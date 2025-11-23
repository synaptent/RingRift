import { evaluateVictory } from '../../src/shared/engine/victoryLogic';
import type { VictoryReason } from '../../src/shared/engine/victoryLogic';
import {
  createTestGameState,
  createTestBoard,
  addMarker,
  addCollapsedSpace,
  getAllBoardPositions,
} from '../utils/fixtures';
import type { GameState } from '../../src/shared/types/game';

describe('shared victory helper – evaluateVictory', () => {
  it('detects ring-elimination victory when eliminatedRings >= victoryThreshold', () => {
    const state = createTestGameState();
    const p1 = state.players[0];

    // Ensure threshold is known and satisfied only by Player 1.
    state.victoryThreshold = 5;
    p1.eliminatedRings = 5;
    state.players[1].eliminatedRings = 0;

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1);
    expect(result.reason as VictoryReason).toBe('ring_elimination');
  });

  it('detects territory-control victory when territorySpaces >= territoryVictoryThreshold', () => {
    const state = createTestGameState();
    const p1 = state.players[0];

    state.territoryVictoryThreshold = 10;
    p1.territorySpaces = 10;
    state.players[1].territorySpaces = 0;

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1);
    expect(result.reason as VictoryReason).toBe('territory_control');
  });

  it('treats bare board with legal re-entry via placement as non-terminal', () => {
    const state = createTestGameState();

    // Bare board: no stacks anywhere.
    state.board.stacks.clear();
    // Players still have rings in hand and no prior progress.
    state.players.forEach((p) => {
      p.ringsInHand = 3;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(false);
    expect(result.winner).toBeUndefined();
    expect(result.reason).toBeUndefined();
  });

  it('applies territory tiebreak on bare-board stalemate when territory differs', () => {
    const state = createTestGameState();

    state.board.stacks.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.eliminatedRings = 0;
      p.territorySpaces = 0;
    });

    // Raise primary thresholds so only the stalemate ladder is exercised.
    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    state.players[0].territorySpaces = 3;
    state.players[1].territorySpaces = 1;

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1);
    expect(result.reason as VictoryReason).toBe('territory_control');
  });

  it('applies eliminated-rings + hand tiebreak on global bare-board stalemate', () => {
    const board = createTestBoard('square8');

    // Make every cell collapsed so that no legal placements exist anywhere.
    getAllBoardPositions(board.type, board.size).forEach((pos) => {
      addCollapsedSpace(board, pos, 1);
    });

    const state: GameState = createTestGameState({ board, boardType: 'square8' });

    state.board.stacks.clear();
    state.players.forEach((p) => {
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    // Player 1 holds more rings in hand; no placements are legal anywhere.
    state.players[0].ringsInHand = 3;
    state.players[1].ringsInHand = 1;

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1);
    expect(result.reason as VictoryReason).toBe('ring_elimination');
    expect(result.handCountsAsEliminated).toBe(true);
  });

  it('applies marker-count tiebreak when territory and eliminated rings are tied', () => {
    const state = createTestGameState();

    state.board.stacks.clear();
    state.board.markers.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 2;
    });

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    // Two markers for Player 1, one for Player 2.
    addMarker(state.board, { x: 0, y: 0 }, 1);
    addMarker(state.board, { x: 1, y: 0 }, 1);
    addMarker(state.board, { x: 0, y: 1 }, 2);

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1);
    expect(result.reason as VictoryReason).toBe('last_player_standing');
  });

  it('falls back to last-actor tiebreak when all other ladders are tied', () => {
    const state = createTestGameState();

    state.board.stacks.clear();
    state.board.markers.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    // With players [1,2] and currentPlayer = 1, the previous player in
    // turn order (Player 2) is treated as the last actor when no history
    // is recorded.
    state.currentPlayer = 1;

    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(2);
    expect(result.reason as VictoryReason).toBe('last_player_standing');
  });
});
