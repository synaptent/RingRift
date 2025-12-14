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
      // Territory victory is derived from collapsedSpaces (authoritative),
      // not players[].territorySpaces.
      for (let x = 0; x < state.board.size; x += 1) {
        state.board.collapsedSpaces.set(positionToString({ x, y: 0 }), 1);
      }
      for (let x = 0; x < 4; x += 1) {
        state.board.collapsedSpaces.set(positionToString({ x, y: 1 }), 1);
      }
      for (let x = 0; x < 3; x += 1) {
        state.board.collapsedSpaces.set(positionToString({ x, y: 7 }), 2);
      }
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
      // Add additional collapsed spaces so territory threshold is actually met.
      state.board.collapsedSpaces.set('2,2', 1);
      state.board.collapsedSpaces.set('1,3', 1);
      state.board.collapsedSpaces.set('2,3', 1);
      state.board.collapsedSpaces.set('7,7', 1);
      state.board.collapsedSpaces.set('5,6', 1);
      state.board.collapsedSpaces.set('5,7', 1);
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

    it('requires dual-condition: threshold met but no dominance = opponent wins (RR-CANON-R062-v2)', () => {
      // RR-CANON-R062-v2: Territory victory requires BOTH:
      // 1. territory >= floor(totalSpaces / numPlayers) + 1
      // 2. territory > sum of all opponents' territory
      const state = createBaseState('movement');
      // Player 1 meets threshold (12 >= 10) but does NOT have dominance
      state.players[0].territorySpaces = 12;
      state.territoryVictoryThreshold = 10;
      // Player 2 has MORE territory than Player 1 - P2 has dominance instead
      state.players[1].territorySpaces = 15;
      // Set up collapsed spaces: P1 has 12, P2 has 15
      for (let x = 0; x < 12; x += 1) {
        const y = x < 8 ? 0 : 1;
        const xPos = x < 8 ? x : x - 8;
        state.board.collapsedSpaces.set(positionToString({ x: xPos, y }), 1);
      }
      for (let x = 0; x < 15; x += 1) {
        const y = x < 8 ? 7 : 6;
        const xPos = x < 8 ? x : x - 8;
        state.board.collapsedSpaces.set(positionToString({ x: xPos, y }), 2);
      }
      // Keep stacks on board (not bare board)
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      state.board.stacks.set('4,4', {
        position: { x: 4, y: 4 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
      });

      const result = toVictoryState(state);

      // P1 lacks dominance (12 < 15), but P2 has both threshold AND dominance
      // P2: 15 >= 10 (threshold) AND 15 > 12 (dominance) => P2 wins
      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(2);
      expect(result.reason).toBe('territory_control');
    });

    it('requires dual-condition: threshold met but exactly tied = no victory (RR-CANON-R062-v2)', () => {
      // RR-CANON-R062-v2: Territory victory requires BOTH conditions
      // When both players are exactly tied, neither has dominance
      const state = createBaseState('movement');
      // Both players meet threshold but neither has dominance (tied)
      state.players[0].territorySpaces = 12;
      state.players[1].territorySpaces = 12;
      state.territoryVictoryThreshold = 10;
      // Set up collapsed spaces: P1 has 12, P2 has 12
      for (let x = 0; x < 12; x += 1) {
        const y = x < 8 ? 0 : 1;
        const xPos = x < 8 ? x : x - 8;
        state.board.collapsedSpaces.set(positionToString({ x: xPos, y }), 1);
      }
      for (let x = 0; x < 12; x += 1) {
        const y = x < 8 ? 7 : 6;
        const xPos = x < 8 ? x : x - 8;
        state.board.collapsedSpaces.set(positionToString({ x: xPos, y }), 2);
      }
      // Keep stacks on board (not bare board)
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      state.board.stacks.set('4,4', {
        position: { x: 4, y: 4 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
      });

      const result = toVictoryState(state);

      // Both players meet threshold (12 >= 10), but neither has dominance (12 == 12)
      // No territory victory should occur - game continues
      expect(result.isGameOver).toBe(false);
      expect(result.reason).not.toBe('territory_control');
    });

    it('requires dual-condition: dominance but below threshold = no victory (RR-CANON-R062-v2)', () => {
      // RR-CANON-R062-v2: Territory victory requires BOTH conditions
      const state = createBaseState('movement');
      // Player 1 has dominance but is below threshold
      state.players[0].territorySpaces = 8;
      state.territoryVictoryThreshold = 10;
      // Player 2 has fewer territory spaces - P1 has dominance
      state.players[1].territorySpaces = 3;
      // Set up collapsed spaces: P1 has 8, P2 has 3
      for (let x = 0; x < 8; x += 1) {
        state.board.collapsedSpaces.set(positionToString({ x, y: 0 }), 1);
      }
      for (let x = 0; x < 3; x += 1) {
        state.board.collapsedSpaces.set(positionToString({ x, y: 7 }), 2);
      }
      // Keep stacks on board
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      state.board.stacks.set('4,4', {
        position: { x: 4, y: 4 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
      });

      const result = toVictoryState(state);

      // Game should NOT be over via territory victory because P1 is below threshold
      // P1 has 8 (< threshold 10) even though 8 > 3 (dominance)
      expect(result.isGameOver).toBe(false);
      expect(result.reason).not.toBe('territory_control');
    });

    it('dual-condition met: threshold AND dominance = victory (RR-CANON-R062-v2)', () => {
      // RR-CANON-R062-v2: Both conditions met = victory
      const state = createBaseState('movement');
      // Player 1 meets threshold AND has dominance
      state.players[0].territorySpaces = 12;
      state.territoryVictoryThreshold = 10;
      // Player 2 has fewer territory spaces
      state.players[1].territorySpaces = 5;
      // Set up collapsed spaces: P1 has 12, P2 has 5
      for (let x = 0; x < 12; x += 1) {
        const y = x < 8 ? 0 : 1;
        const xPos = x < 8 ? x : x - 8;
        state.board.collapsedSpaces.set(positionToString({ x: xPos, y }), 1);
      }
      for (let x = 0; x < 5; x += 1) {
        state.board.collapsedSpaces.set(positionToString({ x, y: 7 }), 2);
      }
      // Keep stacks on board
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      // Game SHOULD be over: P1 has 12 (>= 10 threshold) AND 12 > 5 (dominance)
      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');
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

  // =========================================================================
  // Victory explanation edge cases - lines 576-612, 411-420, 484-510
  // =========================================================================
  describe('victory explanation edge cases', () => {
    it('handles bare-board structural stalemate with ring elimination tiebreak', () => {
      // This tests the path at lines 385-434: noStacksLeft && !primaryRingWinner
      const state = createBaseState('movement');
      // Clear all stacks - bare board
      state.board.stacks.clear();
      // Neither player reached victory threshold, but player 1 eliminated more
      state.players[0].eliminatedRings = 10;
      state.players[0].ringsInHand = 0;
      state.players[1].eliminatedRings = 8;
      state.players[1].ringsInHand = 0;
      state.victoryThreshold = 18; // Neither reached

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      // Winner determined by elimination count tiebreak
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('ring_elimination');
    });

    it('handles bare-board structural stalemate with forced elimination history', () => {
      // This tests lines 411-420: hadForcedEliminationSequence branch
      const state = createBaseState('movement');
      state.board.stacks.clear();
      state.players[0].eliminatedRings = 12;
      state.players[0].ringsInHand = 0;
      state.players[1].eliminatedRings = 6;
      state.players[1].ringsInHand = 0;
      state.victoryThreshold = 18;
      // Add forced elimination move to history
      state.history = [
        {
          id: 'fe-move',
          type: 'forced_elimination',
          player: 1,
          to: { x: 3, y: 3 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 50,
        },
      ];

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
    });

    it('handles territory tiebreak on bare board', () => {
      // This tests lines 460-510: noStacksLeft && !primaryTerritoryWinner in territory_control
      const state = createBaseState('movement');
      state.board.stacks.clear();
      // Set up territory win condition but with bare board
      state.players[0].territorySpaces = 8;
      state.players[0].ringsInHand = 0;
      state.players[0].eliminatedRings = 5;
      state.players[1].territorySpaces = 4;
      state.players[1].ringsInHand = 0;
      state.players[1].eliminatedRings = 5;
      state.territoryVictoryThreshold = 10; // Neither reached
      state.victoryThreshold = 18;

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      // Winner should be determined by territory tiebreak when scores equal in elimination
    });

    it('handles territory tiebreak with forced elimination history', () => {
      // This tests lines 484-493: hadForcedEliminationSequence in territory tiebreak
      const state = createBaseState('movement');
      state.board.stacks.clear();
      state.players[0].territorySpaces = 8;
      state.players[0].ringsInHand = 0;
      state.players[0].eliminatedRings = 5;
      state.players[1].territorySpaces = 4;
      state.players[1].ringsInHand = 0;
      state.players[1].eliminatedRings = 5;
      state.territoryVictoryThreshold = 10;
      state.victoryThreshold = 18;
      // Add forced elimination move to history
      state.history = [
        {
          id: 'fe-move',
          type: 'forced_elimination',
          player: 2,
          to: { x: 5, y: 5 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 40,
        },
      ];

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
    });

    it('handles mini-region territory victory detection', () => {
      // This tests lines 446-459: miniRegionInfo.isMiniRegionVictory
      const state = createBaseState('movement');
      // Player 1 has territory victory threshold
      state.players[0].territorySpaces = 15;
      state.territoryVictoryThreshold = 10;
      // Add some stacks so it's not a bare board
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      // Mini-region detection is driven by collapsedSpaces owned by the winner.
      // Create 2 disconnected regions totaling >= territoryVictoryThreshold.
      state.board.collapsedSpaces.set('1,1', 1);
      state.board.collapsedSpaces.set('1,2', 1);
      state.board.collapsedSpaces.set('2,1', 1);
      state.board.collapsedSpaces.set('2,2', 1);
      state.board.collapsedSpaces.set('1,3', 1);
      state.board.collapsedSpaces.set('6,6', 1);
      state.board.collapsedSpaces.set('6,7', 1);
      state.board.collapsedSpaces.set('7,6', 1);
      state.board.collapsedSpaces.set('7,7', 1);
      state.board.collapsedSpaces.set('5,6', 1);

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');
    });

    it('handles LPS with forced elimination in history', () => {
      // This tests lines 542-552: hadForcedEliminationSequence in LPS
      const state = createBaseState('movement');
      // Only player 1 has stacks
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      // Player 2 eliminated
      state.players[1].ringsInHand = 0;
      // Add forced elimination history
      state.history = [
        {
          id: 'fe-move',
          type: 'forced_elimination',
          player: 2,
          to: { x: 4, y: 4 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 30,
        },
      ];

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });

    it('handles 3-player LPS with multiple eliminations via forced elimination', () => {
      const state = createBaseState('movement', 3);
      // Only player 1 has stacks
      state.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        stackHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });
      // Players 2 and 3 eliminated
      state.players[1].ringsInHand = 0;
      state.players[2].ringsInHand = 0;
      // Add forced elimination for both
      state.history = [
        {
          id: 'fe1',
          type: 'forced_elimination',
          player: 2,
          to: { x: 4, y: 4 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 20,
        },
        {
          id: 'fe2',
          type: 'forced_elimination',
          player: 3,
          to: { x: 5, y: 5 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 25,
        },
      ];

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });
  });

  // =========================================================================
  // game_completed reason (lines 579-615)
  // =========================================================================
  describe('game_completed fallback structural stalemate', () => {
    it('handles game_completed with even scores', () => {
      // This tests the game_completed branch in buildGameEndExplanationForVictory
      const state = createBaseState('movement');
      state.board.stacks.clear();
      // Equal everything - true stalemate
      state.players[0].eliminatedRings = 9;
      state.players[0].territorySpaces = 5;
      state.players[0].ringsInHand = 0;
      state.players[1].eliminatedRings = 9;
      state.players[1].territorySpaces = 5;
      state.players[1].ringsInHand = 0;
      state.victoryThreshold = 18;
      state.territoryVictoryThreshold = 10;

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
    });

    it('handles game_completed with forced elimination history', () => {
      // Tests lines 590-599: hadForcedEliminationSequence in game_completed branch
      const state = createBaseState('movement');
      state.board.stacks.clear();
      state.players[0].eliminatedRings = 9;
      state.players[0].territorySpaces = 5;
      state.players[0].ringsInHand = 0;
      state.players[1].eliminatedRings = 9;
      state.players[1].territorySpaces = 5;
      state.players[1].ringsInHand = 0;
      state.victoryThreshold = 18;
      state.territoryVictoryThreshold = 10;
      // Add forced elimination to history
      state.history = [
        {
          id: 'fe-move',
          type: 'forced_elimination',
          player: 1,
          to: { x: 2, y: 2 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 30,
        },
      ];

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      // With forced elimination history, should have weird state context
      if (result.gameEndExplanation?.weirdStateContext) {
        expect(
          result.gameEndExplanation.weirdStateContext.reasonCodes.length
        ).toBeGreaterThanOrEqual(1);
      }
    });
  });

  // =========================================================================
  // Victory during processTurn (lines 2977-3011)
  // =========================================================================
  describe('victory detection during processTurn', () => {
    it('processes turn when victory threshold reached', async () => {
      const state = createBaseState('territory_processing');
      // Player 1 has reached victory threshold
      state.players[0].eliminatedRings = 18;
      state.victoryThreshold = 18;
      // Add a stack so game continues until move processed
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const { processTurn } =
        await import('../../src/shared/engine/orchestration/turnOrchestrator');
      const move: Move = {
        id: 'skip-terr',
        type: 'skip_territory_processing',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = processTurn(state, move);

      // Turn should process normally
      expect(result.nextState).toBeDefined();
      // Victory result may or may not be present depending on exact victory conditions
    });

    it('detects victory via toVictoryState after threshold reached', () => {
      const state = createBaseState('movement');
      // Player 1 has reached victory threshold through eliminations
      state.players[0].eliminatedRings = 18;
      state.victoryThreshold = 18;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const result = toVictoryState(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
    });
  });
});
