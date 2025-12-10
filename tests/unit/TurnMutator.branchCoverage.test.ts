/**
 * TurnMutator.branchCoverage.test.ts
 *
 * Branch coverage tests for TurnMutator.ts
 */

import { mutateTurnChange, mutatePhaseChange } from '../../src/shared/engine/mutators/TurnMutator';
import type { GameState, GamePhase, Player } from '../../src/shared/engine/types';

describe('TurnMutator branch coverage', () => {
  const createPlayer = (playerNumber: number): Player => ({
    id: `player-${playerNumber}`,
    username: `Player ${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
  });

  const createBaseState = (numPlayers: number = 2): GameState => ({
    id: 'test-game',
    currentPlayer: 1,
    currentPhase: 'movement' as GamePhase,
    gameStatus: 'active',
    boardType: 'square8',
    players: Array.from({ length: numPlayers }, (_, i) => createPlayer(i + 1)),
    board: {
      type: 'square8',
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      territories: new Map(),
      formedLines: [],
      collapsedSpaces: new Map(),
      eliminatedRings: {},
    },
    moveHistory: [],
    history: [],
    lastMoveAt: new Date(),
    createdAt: new Date(),
    isRated: false,
    spectators: [],
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
    maxPlayers: numPlayers,
    totalRingsInPlay: 36,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
  });

  describe('mutateTurnChange', () => {
    it('rotates to next player in 2-player game', () => {
      const state = createBaseState(2);
      state.currentPlayer = 1;

      const newState = mutateTurnChange(state);

      expect(newState.currentPlayer).toBe(2);
    });

    it('wraps around to player 1 after last player in 2-player game', () => {
      const state = createBaseState(2);
      state.currentPlayer = 2;

      const newState = mutateTurnChange(state);

      expect(newState.currentPlayer).toBe(1);
    });

    it('rotates correctly in 3-player game', () => {
      const state = createBaseState(3);
      state.currentPlayer = 2;

      const newState = mutateTurnChange(state);

      expect(newState.currentPlayer).toBe(3);
    });

    it('wraps around to player 1 after last player in 3-player game', () => {
      const state = createBaseState(3);
      state.currentPlayer = 3;

      const newState = mutateTurnChange(state);

      expect(newState.currentPlayer).toBe(1);
    });

    it('rotates correctly in 4-player game', () => {
      const state = createBaseState(4);
      state.currentPlayer = 3;

      const newState = mutateTurnChange(state);

      expect(newState.currentPlayer).toBe(4);
    });

    it('wraps around in 4-player game', () => {
      const state = createBaseState(4);
      state.currentPlayer = 4;

      const newState = mutateTurnChange(state);

      expect(newState.currentPlayer).toBe(1);
    });

    it('resets phase to ring_placement', () => {
      const state = createBaseState(2);
      state.currentPhase = 'movement';

      const newState = mutateTurnChange(state);

      expect(newState.currentPhase).toBe('ring_placement');
    });

    it('updates lastMoveAt timestamp', () => {
      const state = createBaseState(2);
      const originalTime = new Date(2024, 0, 1);
      state.lastMoveAt = originalTime;

      const newState = mutateTurnChange(state);

      expect(newState.lastMoveAt).not.toBe(originalTime);
      expect(newState.lastMoveAt.getTime()).toBeGreaterThan(originalTime.getTime());
    });

    it('preserves other state fields', () => {
      const state = createBaseState(2);
      state.gameStatus = 'active';
      state.isRated = true;

      const newState = mutateTurnChange(state);

      expect(newState.gameStatus).toBe('active');
      expect(newState.isRated).toBe(true);
      expect(newState.boardType).toBe('square8');
    });

    it('creates shallow copies of players array', () => {
      const state = createBaseState(2);

      const newState = mutateTurnChange(state);

      expect(newState.players).not.toBe(state.players);
      expect(newState.players[0]).not.toBe(state.players[0]);
    });

    it('creates shallow copy of moveHistory', () => {
      const state = createBaseState(2);
      state.moveHistory = [{ id: 'move-1', type: 'place_ring', player: 1 } as never];

      const newState = mutateTurnChange(state);

      expect(newState.moveHistory).not.toBe(state.moveHistory);
      expect(newState.moveHistory.length).toBe(1);
    });
  });

  describe('mutatePhaseChange', () => {
    it('changes phase to movement', () => {
      const state = createBaseState(2);
      state.currentPhase = 'ring_placement';

      const newState = mutatePhaseChange(state, 'movement');

      expect(newState.currentPhase).toBe('movement');
    });

    it('changes phase to capture', () => {
      const state = createBaseState(2);
      state.currentPhase = 'movement';

      const newState = mutatePhaseChange(state, 'capture');

      expect(newState.currentPhase).toBe('capture');
    });

    it('changes phase to chain_capture', () => {
      const state = createBaseState(2);
      state.currentPhase = 'capture';

      const newState = mutatePhaseChange(state, 'chain_capture');

      expect(newState.currentPhase).toBe('chain_capture');
    });

    it('changes phase to line_processing', () => {
      const state = createBaseState(2);
      state.currentPhase = 'movement';

      const newState = mutatePhaseChange(state, 'line_processing');

      expect(newState.currentPhase).toBe('line_processing');
    });

    it('changes phase to territory_processing', () => {
      const state = createBaseState(2);
      state.currentPhase = 'line_processing';

      const newState = mutatePhaseChange(state, 'territory_processing');

      expect(newState.currentPhase).toBe('territory_processing');
    });

    it('changes phase to ring_placement', () => {
      const state = createBaseState(2);
      state.currentPhase = 'territory_processing';

      const newState = mutatePhaseChange(state, 'ring_placement');

      expect(newState.currentPhase).toBe('ring_placement');
    });

    it('updates lastMoveAt timestamp', () => {
      const state = createBaseState(2);
      const originalTime = new Date(2024, 0, 1);
      state.lastMoveAt = originalTime;

      const newState = mutatePhaseChange(state, 'movement');

      expect(newState.lastMoveAt).not.toBe(originalTime);
      expect(newState.lastMoveAt.getTime()).toBeGreaterThan(originalTime.getTime());
    });

    it('preserves currentPlayer', () => {
      const state = createBaseState(2);
      state.currentPlayer = 2;

      const newState = mutatePhaseChange(state, 'movement');

      expect(newState.currentPlayer).toBe(2);
    });

    it('preserves other state fields', () => {
      const state = createBaseState(2);
      state.gameStatus = 'active';
      state.isRated = true;
      state.id = 'my-game-id';

      const newState = mutatePhaseChange(state, 'movement');

      expect(newState.gameStatus).toBe('active');
      expect(newState.isRated).toBe(true);
      expect(newState.id).toBe('my-game-id');
    });

    it('creates shallow copy of moveHistory', () => {
      const state = createBaseState(2);
      state.moveHistory = [{ id: 'move-1', type: 'place_ring', player: 1 } as never];

      const newState = mutatePhaseChange(state, 'movement');

      expect(newState.moveHistory).not.toBe(state.moveHistory);
      expect(newState.moveHistory.length).toBe(1);
    });
  });
});
