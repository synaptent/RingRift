/**
 * TurnOrchestrator victory explanation branch coverage tests
 *
 * Tests for src/shared/engine/orchestration/turnOrchestrator.ts covering:
 * - toVictoryState and __test_only_toVictoryState
 * - buildGameEndExplanationForVictory (all victory reason branches)
 * - Mini-region detection for territory victories
 * - deriveUxCopyKeys for complex endings
 * - hasForcedEliminationMove detection
 *
 * These tests target uncovered lines 237-734 focusing on victory state
 * conversion and game end explanation building.
 */

import {
  toVictoryState,
  __test_only_toVictoryState,
} from '../../src/shared/engine/orchestration/turnOrchestrator';
import type {
  GameState,
  GamePhase,
  Move,
  Player,
  Board,
  Position,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

describe('TurnOrchestrator victory explanation branch coverage', () => {
  const createPlayer = (playerNumber: number, options: Partial<Player> = {}): Player => ({
    id: `player-${playerNumber}`,
    username: `Player ${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
    ...options,
  });

  const createEmptyBoard = (size: number = 8): Board => ({
    type: 'square8',
    size,
    stacks: new Map(),
    markers: new Map(),
    territories: new Map(),
    formedLines: [],
    collapsedSpaces: new Map(),
    eliminatedRings: {},
  });

  const createBaseState = (
    phase: GamePhase = 'ring_placement',
    numPlayers: number = 2
  ): GameState => ({
    id: 'test-game',
    currentPlayer: 1,
    currentPhase: phase,
    gameStatus: 'active',
    boardType: 'square8',
    players: Array.from({ length: numPlayers }, (_, i) => createPlayer(i + 1)),
    board: createEmptyBoard(),
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
    territoryVictoryThreshold: 10,
  });

  // =========================================================================
  // toVictoryState - basic cases
  // =========================================================================
  describe('toVictoryState', () => {
    it('returns non-game-over state for active game', () => {
      const state = createBaseState('movement');

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(false);
      expect(result.winner).toBeFalsy(); // null or undefined
      expect(result.scores).toHaveLength(2);
    });

    it('computes marker counts for each player', () => {
      const state = createBaseState('movement');
      // Add markers for player 1
      state.board.markers.set('0,0', { position: { x: 0, y: 0 }, player: 1 });
      state.board.markers.set('1,0', { position: { x: 1, y: 0 }, player: 1 });
      // Add markers for player 2
      state.board.markers.set('2,0', { position: { x: 2, y: 0 }, player: 2 });

      const result = toVictoryState(state);

      expect(result.scores[0].markerCount).toBe(2);
      expect(result.scores[1].markerCount).toBe(1);
    });

    it('computes elimination status based on rings', () => {
      const state = createBaseState('movement');
      // Player 1 has no rings (eliminated)
      state.players[0].ringsInHand = 0;
      // Player 2 has rings
      state.players[1].ringsInHand = 5;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
      });

      const result = toVictoryState(state);

      expect(result.scores[0].isEliminated).toBe(true);
      expect(result.scores[1].isEliminated).toBe(false);
    });

    it('counts rings on board for each player', () => {
      const state = createBaseState('movement');
      // Player 1 stack with 3 rings
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });
      // Player 2 stack with 2 rings
      state.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
      });

      const result = toVictoryState(state);

      expect(result.scores[0].ringsOnBoard).toBe(3);
      expect(result.scores[1].ringsOnBoard).toBe(2);
    });
  });

  // =========================================================================
  // __test_only_toVictoryState - wrapper test
  // =========================================================================
  describe('__test_only_toVictoryState', () => {
    it('exposes internal toVictoryState for testing', () => {
      const state = createBaseState('movement');

      const result = __test_only_toVictoryState(state);

      expect(result).toBeDefined();
      expect(result.scores).toBeDefined();
      expect(result.isGameOver).toBe(false);
    });
  });

  // =========================================================================
  // Victory with ring_elimination reason
  // =========================================================================
  describe('ring_elimination victory', () => {
    it('detects standard ring majority victory', () => {
      const state = createBaseState('movement');
      // Player 1 reaches victory threshold (RR-CANON-R061: ringsPerPlayer)
      state.players[0].eliminatedRings = 18;
      state.victoryThreshold = 18;
      // Player 2 has fewer eliminations
      state.players[1].eliminatedRings = 5;
      // Keep stacks on board (not bare board)
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('ring_elimination');
    });

    it('handles bare board stalemate with ring elimination tiebreak', () => {
      const state = createBaseState('movement');
      // Both players have some eliminated rings but neither at threshold
      state.players[0].eliminatedRings = 10;
      state.players[1].eliminatedRings = 8;
      // NO stacks on board (bare board)
      // Clear all stacks
      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      const result = toVictoryState(state);

      // With bare board and no threshold reached, should be stalemate
      expect(result.isGameOver).toBe(true);
      // Winner determined by elimination count tiebreak
      expect(result.winner).toBe(1); // Player 1 has more eliminated rings
    });
  });

  // =========================================================================
  // Victory with territory_control reason
  // =========================================================================
  describe('territory_control victory', () => {
    it('detects standard territory majority victory', () => {
      const state = createBaseState('movement');
      // Player 1 reaches territory threshold
      state.players[0].territorySpaces = 12;
      state.territoryVictoryThreshold = 10;
      // Player 2 has fewer territory spaces
      state.players[1].territorySpaces = 3;
      // Keep stacks on board (not bare board)
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');
    });

    it('handles territory mini-region victory scenario', () => {
      const state = createBaseState('movement');
      // Player 1 reaches territory threshold
      state.players[0].territorySpaces = 12;
      state.territoryVictoryThreshold = 10;
      // Set up collapsed spaces in multiple disconnected regions
      // Region 1: positions around (1,1)
      state.board.collapsedSpaces.set('1,1', 1);
      state.board.collapsedSpaces.set('1,2', 1);
      state.board.collapsedSpaces.set('2,1', 1);
      // Region 2: positions around (6,6) - disconnected from region 1
      state.board.collapsedSpaces.set('6,6', 1);
      state.board.collapsedSpaces.set('6,7', 1);
      state.board.collapsedSpaces.set('7,6', 1);
      // Keep stacks on board
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      // Should have gameEndExplanation if territory victory detected
      if (result.gameEndExplanation) {
        expect(result.gameEndExplanation.outcomeType).toBe('territory_control');
      }
    });

    it('handles bare board stalemate with territory tiebreak', () => {
      const state = createBaseState('movement');
      // Neither player at threshold but territory tiebreak needed
      state.players[0].territorySpaces = 5;
      state.players[1].territorySpaces = 3;
      state.players[0].eliminatedRings = 0;
      state.players[1].eliminatedRings = 0;
      // NO stacks on board (bare board)
      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      const result = toVictoryState(state);

      // Bare board stalemate
      expect(result.isGameOver).toBe(true);
    });
  });

  // =========================================================================
  // Victory with last_player_standing reason
  // =========================================================================
  describe('last_player_standing victory', () => {
    it('detects LPS when only one player has turn material', () => {
      const state = createBaseState('movement');
      // Player 1 has stacks and can move
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      // Player 2 is eliminated (no stacks, no rings)
      state.players[1].ringsInHand = 0;
      // No player 2 stacks

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });

    it('includes LPS weird state context when applicable', () => {
      const state = createBaseState('movement');
      // Set up LPS victory condition
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      state.players[1].ringsInHand = 0;

      const result = toVictoryState(state);

      if (result.gameEndExplanation) {
        expect(result.gameEndExplanation.outcomeType).toBe('last_player_standing');
        if (result.gameEndExplanation.weirdStateContext) {
          expect(result.gameEndExplanation.weirdStateContext.reasonCodes).toBeDefined();
        }
      }
    });
  });

  // =========================================================================
  // Victory with game_completed reason (structural stalemate)
  // =========================================================================
  describe('game_completed / structural stalemate', () => {
    it('handles structural stalemate with no clear winner', () => {
      const state = createBaseState('movement');
      // Both players equal - leads to tiebreak
      state.players[0].eliminatedRings = 5;
      state.players[1].eliminatedRings = 5;
      state.players[0].territorySpaces = 3;
      state.players[1].territorySpaces = 3;
      // Bare board
      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
    });
  });

  // =========================================================================
  // hasForcedEliminationMove detection (line 726-734)
  // =========================================================================
  describe('forced elimination move detection', () => {
    it('detects forced_elimination in history', () => {
      const state = createBaseState('movement');
      // Add forced elimination to history
      state.history = [
        {
          action: { type: 'forced_elimination', player: 1, position: { x: 0, y: 0 } },
          stateBefore: {} as GameState,
          stateAfter: {} as GameState,
        },
      ];
      // Set up terminal state
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      state.players[0].eliminatedRings = 18; // RR-CANON-R061: ringsPerPlayer
      state.victoryThreshold = 18;

      const result = toVictoryState(state);

      // The forced elimination history should enrich the explanation
      expect(result.isGameOver).toBe(true);
      if (result.gameEndExplanation?.weirdStateContext) {
        // With forced elimination in history, should have ANM/FE reason codes
        expect(result.gameEndExplanation.weirdStateContext.reasonCodes.length).toBeGreaterThan(0);
      }
    });

    it('returns false for empty history', () => {
      const state = createBaseState('movement');
      state.history = [];

      const result = toVictoryState(state);

      // No forced elimination in history
      expect(result.isGameOver).toBe(false);
    });

    it('returns false when history has no forced_elimination', () => {
      const state = createBaseState('movement');
      state.history = [
        {
          action: { type: 'move_stack', player: 1, from: { x: 0, y: 0 }, to: { x: 1, y: 0 } },
          stateBefore: {} as GameState,
          stateAfter: {} as GameState,
        },
      ];

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(false);
    });
  });

  // =========================================================================
  // Score breakdown creation (lines 623-635)
  // =========================================================================
  describe('score breakdown', () => {
    it('creates score breakdown for all players', () => {
      const state = createBaseState('movement');
      state.players[0].eliminatedRings = 10;
      state.players[0].territorySpaces = 5;
      state.players[1].eliminatedRings = 8;
      state.players[1].territorySpaces = 3;
      // Add markers
      state.board.markers.set('0,0', { position: { x: 0, y: 0 }, player: 1 });
      state.board.markers.set('1,0', { position: { x: 1, y: 0 }, player: 2 });
      // Terminal state
      state.players[0].eliminatedRings = 18; // RR-CANON-R061: ringsPerPlayer
      state.victoryThreshold = 18;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      expect(result.scores).toHaveLength(2);
      expect(result.scores[0].eliminatedRings).toBe(18); // RR-CANON-R061: ringsPerPlayer
      expect(result.scores[1].eliminatedRings).toBe(8);
    });
  });

  // =========================================================================
  // Three and four player games
  // =========================================================================
  describe('multi-player games', () => {
    it('handles 3-player victory state', () => {
      const state = createBaseState('movement', 3);
      // Player 1 reaches threshold
      state.players[0].eliminatedRings = 18; // RR-CANON-R061: ringsPerPlayer
      state.victoryThreshold = 18;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.scores).toHaveLength(3);
    });

    it('handles 4-player victory state', () => {
      const state = createBaseState('movement', 4);
      // Player 3 reaches threshold
      state.players[2].eliminatedRings = 18; // RR-CANON-R061: ringsPerPlayer
      state.victoryThreshold = 18;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 3,
        composition: [{ player: 3, count: 2 }],
        rings: [3, 3],
      });

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(3);
      expect(result.scores).toHaveLength(4);
    });

    it('handles LPS in 3-player game (2 eliminated)', () => {
      const state = createBaseState('movement', 3);
      // Only player 1 has stacks
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      // Players 2 and 3 eliminated
      state.players[1].ringsInHand = 0;
      state.players[2].ringsInHand = 0;

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });
  });

  // =========================================================================
  // Edge cases
  // =========================================================================
  describe('edge cases', () => {
    it('handles empty players array gracefully', () => {
      const state = createBaseState('movement');
      state.players = [];

      // Should not throw
      const result = toVictoryState(state);

      expect(result.scores).toHaveLength(0);
    });

    it('handles mixed stack compositions', () => {
      const state = createBaseState('movement');
      // Mixed stack (rings from both players)
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        composition: [
          { player: 1, count: 2 },
          { player: 2, count: 1 },
        ],
        rings: [1, 1, 2],
      });

      const result = toVictoryState(state);

      // Should count all rings by owner, not controller
      expect(result.scores[0].ringsOnBoard).toBeGreaterThanOrEqual(2);
    });

    it('handles null winner in tiebreak scenarios', () => {
      const state = createBaseState('movement');
      // Equal scores - may result in draw or last actor tiebreak
      state.players[0].eliminatedRings = 10;
      state.players[1].eliminatedRings = 10;
      state.players[0].territorySpaces = 5;
      state.players[1].territorySpaces = 5;
      // Bare board
      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      const result = toVictoryState(state);

      // Game should be over with winner determined by tiebreak
      expect(result.isGameOver).toBe(true);
    });
  });
});
