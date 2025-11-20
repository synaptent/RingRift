import { BoardType, GameState } from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { checkSandboxVictory } from '../../src/client/sandbox/sandboxVictory';
import { createTestBoard, createTestGameState } from '../utils/fixtures';

/**
 * Victory parity tests: RuleEngine vs Client sandbox.
 *
 * These tests construct synthetic structural-stalemate states and assert that
 * the backend RuleEngine and the client-local sandbox victory helper choose
 * the same winner and reason for the final game result.
 *
 * Rules references:
 * - ringrift_complete_rules.md §13.1–13.4, 16.9.4.5 (victory & stalemate ladder)
 * - FAQ 11, 18, 21, 24 (stalemate, thresholds, player-count examples)
 */

describe('Victory parity: RuleEngine vs sandbox stalemate ladder (square8 / 2p)', () => {
  const boardType: BoardType = 'square8';

  function makeStalemateBase(): GameState {
    const base = createTestGameState({ boardType, board: createTestBoard(boardType) });

    // Structural terminality: no stacks and no rings in hand for any player.
    base.board.stacks.clear();
    base.board.markers.clear();
    base.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    // Keep thresholds high enough that we do not trigger primary
    // ring/territory victories in these tests.
    base.victoryThreshold = 1000;
    base.territoryVictoryThreshold = 1000;

    return base;
  }

  function getBackendResult(state: GameState) {
    const bm = new BoardManager(boardType);
    const engine = new RuleEngine(bm as any, boardType as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (engine as any).checkGameEnd(state) as { isGameOver: boolean; winner?: number; reason?: string };
  }

  it('parity_territory_tiebreak', () => {
    const state = makeStalemateBase();

    // Player 1 has more territory; eliminated rings and markers tied.
    state.players[0].territorySpaces = 3; // P1
    state.players[1].territorySpaces = 1; // P2

    const backend = getBackendResult(state);
    const sandbox = checkSandboxVictory(state);

    expect(backend.isGameOver).toBe(true);
    expect(sandbox).not.toBeNull();

    expect(backend.winner).toBe(1);
    expect(sandbox!.winner).toBe(1);
    expect(backend.reason).toBe('territory_control');
    expect(sandbox!.reason).toBe('territory_control');
  });

  it('parity_eliminated_rings_tiebreak', () => {
    const state = makeStalemateBase();

    // Territory tied; Player 1 has more eliminated rings.
    state.players[0].eliminatedRings = 4; // P1
    state.players[1].eliminatedRings = 2; // P2
    state.totalRingsEliminated = 6;

    const backend = getBackendResult(state);
    const sandbox = checkSandboxVictory(state);

    expect(backend.isGameOver).toBe(true);
    expect(sandbox).not.toBeNull();

    expect(backend.winner).toBe(1);
    expect(sandbox!.winner).toBe(1);
    expect(backend.reason).toBe('ring_elimination');
    expect(sandbox!.reason).toBe('ring_elimination');
  });

  it('parity_markers_tiebreak', () => {
    const state = makeStalemateBase();

    // Territory and eliminated rings tied; Player 1 has more markers.
    state.players[0].eliminatedRings = 2;
    state.players[1].eliminatedRings = 2;

    // Two markers for P1, one for P2.
    state.board.markers.set('0,0', {
      player: 1,
      position: { x: 0, y: 0 },
      type: 'regular',
    });
    state.board.markers.set('1,0', {
      player: 1,
      position: { x: 1, y: 0 },
      type: 'regular',
    });
    state.board.markers.set('0,1', {
      player: 2,
      position: { x: 0, y: 1 },
      type: 'regular',
    });

    const backend = getBackendResult(state);
    const sandbox = checkSandboxVictory(state);

    expect(backend.isGameOver).toBe(true);
    expect(sandbox).not.toBeNull();

    expect(backend.winner).toBe(1);
    expect(sandbox!.winner).toBe(1);
    expect(backend.reason).toBe('last_player_standing');
    expect(sandbox!.reason).toBe('last_player_standing');
  });

  it('parity_last_actor_final_rung', () => {
    const state = makeStalemateBase();

    // Everything tied at 0: territory, eliminated rings, markers.
    state.players.forEach((p) => {
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });
    state.board.markers.clear();

    // With players [1,2] and currentPlayer = 1, both backends treat
    // Player 2 as the last actor when no history is recorded.
    state.currentPlayer = 1;

    const backend = getBackendResult(state);
    const sandbox = checkSandboxVictory(state);

    expect(backend.isGameOver).toBe(true);
    expect(sandbox).not.toBeNull();

    expect(backend.winner).toBe(2);
    expect(sandbox!.winner).toBe(2);
    expect(backend.reason).toBe('last_player_standing');
    expect(sandbox!.reason).toBe('last_player_standing');
  });
});
