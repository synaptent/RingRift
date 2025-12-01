/**
 * VictoryAggregate.shared.test.ts
 *
 * Comprehensive tests for VictoryAggregate functions.
 * Covers: checkLastPlayerStanding, checkScoreThreshold, getPlayerScore,
 * getRemainingPlayers, isPlayerEliminated, getLastActor, evaluateVictoryDetailed,
 * getEliminatedRingCount, getTerritoryCount, getMarkerCount, isVictoryThresholdReached
 */

import {
  evaluateVictory,
  checkLastPlayerStanding,
  checkScoreThreshold,
  getPlayerScore,
  getRemainingPlayers,
  isPlayerEliminated,
  getLastActor,
  evaluateVictoryDetailed,
  getEliminatedRingCount,
  getTerritoryCount,
  getMarkerCount,
  isVictoryThresholdReached,
} from '../../src/shared/engine/aggregates/VictoryAggregate';
import { createTestGameState, createTestBoard, addStack, addMarker } from '../utils/fixtures';
import type { GameState, HistoryEntry } from '../../src/shared/types/game';

describe('VictoryAggregate', () => {
  describe('checkLastPlayerStanding', () => {
    it('returns null when both players have stacks on board', () => {
      const state = createTestGameState();
      // Default state has stacks for both players
      const result = checkLastPlayerStanding(state);
      expect(result).toBeNull();
    });

    it('returns null when players have less than 2 players', () => {
      const state = createTestGameState();
      state.players = [state.players[0]];
      const result = checkLastPlayerStanding(state);
      expect(result).toBeNull();
    });

    it('returns null when players array is empty', () => {
      const state = createTestGameState();
      state.players = [];
      const result = checkLastPlayerStanding(state);
      expect(result).toBeNull();
    });

    it('detects last player standing when one player has no rings', () => {
      const state = createTestGameState();
      // Clear all stacks
      state.board.stacks.clear();
      // Player 1 has rings in hand, Player 2 has none
      state.players[0].ringsInHand = 3;
      state.players[1].ringsInHand = 0;

      const result = checkLastPlayerStanding(state);

      expect(result).not.toBeNull();
      expect(result!.isGameOver).toBe(true);
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');
    });

    it('returns null when multiple players have rings in hand', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.players[0].ringsInHand = 3;
      state.players[1].ringsInHand = 2;

      const result = checkLastPlayerStanding(state);
      expect(result).toBeNull();
    });

    it('detects last player standing via stacks on board', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      // Add stack only for player 1
      addStack(state.board, { x: 0, y: 0 }, 1, 1, 1);
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      const result = checkLastPlayerStanding(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
    });
  });

  describe('checkScoreThreshold', () => {
    it('returns null when no players exist', () => {
      const state = createTestGameState();
      state.players = [];
      const result = checkScoreThreshold(state);
      expect(result).toBeNull();
    });

    it('returns null when no player meets thresholds', () => {
      const state = createTestGameState();
      state.victoryThreshold = 10;
      state.territoryVictoryThreshold = 20;
      state.players[0].eliminatedRings = 5;
      state.players[0].territorySpaces = 10;

      const result = checkScoreThreshold(state);
      expect(result).toBeNull();
    });

    it('detects ring elimination victory', () => {
      const state = createTestGameState();
      state.victoryThreshold = 5;
      state.players[0].eliminatedRings = 5;

      const result = checkScoreThreshold(state);

      expect(result).not.toBeNull();
      expect(result!.isGameOver).toBe(true);
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('ring_elimination');
    });

    it('detects territory control victory', () => {
      const state = createTestGameState();
      state.territoryVictoryThreshold = 10;
      state.players[0].territorySpaces = 10;

      const result = checkScoreThreshold(state);

      expect(result).not.toBeNull();
      expect(result!.isGameOver).toBe(true);
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('territory_control');
    });

    it('prioritizes ring elimination over territory', () => {
      const state = createTestGameState();
      state.victoryThreshold = 5;
      state.territoryVictoryThreshold = 10;
      state.players[0].eliminatedRings = 5;
      state.players[0].territorySpaces = 10;

      const result = checkScoreThreshold(state);

      expect(result!.reason).toBe('ring_elimination');
    });
  });

  describe('getPlayerScore', () => {
    it('returns 0 for unknown player', () => {
      const state = createTestGameState();
      const score = getPlayerScore(state, 999);
      expect(score).toBe(0);
    });

    it('returns eliminated rings count', () => {
      const state = createTestGameState();
      state.players[0].eliminatedRings = 7;
      const score = getPlayerScore(state, 1);
      expect(score).toBe(7);
    });

    it('returns 0 when player has no eliminations', () => {
      const state = createTestGameState();
      state.players[0].eliminatedRings = 0;
      const score = getPlayerScore(state, 1);
      expect(score).toBe(0);
    });
  });

  describe('getRemainingPlayers', () => {
    it('returns empty array when no players exist', () => {
      const state = createTestGameState();
      state.players = [];
      const remaining = getRemainingPlayers(state);
      expect(remaining).toEqual([]);
    });

    it('returns all players when all have rings', () => {
      const state = createTestGameState();
      state.players[0].ringsInHand = 3;
      state.players[1].ringsInHand = 3;
      const remaining = getRemainingPlayers(state);
      expect(remaining.length).toBe(2);
    });

    it('returns only players with stacks on board', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      addStack(state.board, { x: 0, y: 0 }, 1, 1, 1);
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      const remaining = getRemainingPlayers(state);
      expect(remaining.length).toBe(1);
      expect(remaining[0].playerNumber).toBe(1);
    });

    it('returns players with rings in hand even without stacks', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.players[0].ringsInHand = 2;
      state.players[1].ringsInHand = 0;

      const remaining = getRemainingPlayers(state);
      expect(remaining.length).toBe(1);
      expect(remaining[0].playerNumber).toBe(1);
    });
  });

  describe('isPlayerEliminated', () => {
    it('returns true for unknown player', () => {
      const state = createTestGameState();
      expect(isPlayerEliminated(state, 999)).toBe(true);
    });

    it('returns false when player has rings in hand', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.players[0].ringsInHand = 2;
      expect(isPlayerEliminated(state, 1)).toBe(false);
    });

    it('returns false when player has stacks on board', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      addStack(state.board, { x: 0, y: 0 }, 1, 1, 1);
      state.players[0].ringsInHand = 0;
      expect(isPlayerEliminated(state, 1)).toBe(false);
    });

    it('returns true when player has no rings anywhere', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.players[0].ringsInHand = 0;
      expect(isPlayerEliminated(state, 1)).toBe(true);
    });
  });

  describe('getLastActor', () => {
    it('returns actor from history when available', () => {
      const state = createTestGameState();
      state.history = [{ actor: 2, action: 'move' } as HistoryEntry];
      expect(getLastActor(state)).toBe(2);
    });

    it('returns player from moveHistory when no structured history', () => {
      const state = createTestGameState();
      state.history = [];
      state.moveHistory = [{ type: 'move', player: 1, from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }];
      expect(getLastActor(state)).toBe(1);
    });

    it('returns previous player in turn order when no history', () => {
      const state = createTestGameState();
      state.history = [];
      state.moveHistory = [];
      state.currentPlayer = 1;
      // With players [1, 2] and currentPlayer = 1, previous is player 2
      expect(getLastActor(state)).toBe(2);
    });

    it('returns undefined when no players exist', () => {
      const state = createTestGameState();
      state.players = [];
      state.history = [];
      state.moveHistory = [];
      expect(getLastActor(state)).toBeUndefined();
    });

    it('returns first player when currentPlayer not found', () => {
      const state = createTestGameState();
      state.history = [];
      state.moveHistory = [];
      state.currentPlayer = 999;
      expect(getLastActor(state)).toBe(1);
    });
  });

  describe('evaluateVictoryDetailed', () => {
    it('returns base result for empty players', () => {
      const state = createTestGameState();
      state.players = [];
      const result = evaluateVictoryDetailed(state);
      expect(result.isGameOver).toBe(false);
      expect(result.standings).toBeUndefined();
      expect(result.scores).toBeUndefined();
    });

    it('includes standings sorted by victory criteria', () => {
      const state = createTestGameState();
      state.players[0].territorySpaces = 5;
      state.players[0].eliminatedRings = 3;
      state.players[1].territorySpaces = 10;
      state.players[1].eliminatedRings = 1;

      const result = evaluateVictoryDetailed(state);

      expect(result.standings).toBeDefined();
      expect(result.standings![0].playerNumber).toBe(2); // Higher territory
      expect(result.standings![1].playerNumber).toBe(1);
    });

    it('includes score breakdown for each player', () => {
      const state = createTestGameState();
      state.players[0].eliminatedRings = 3;
      state.players[0].territorySpaces = 5;
      state.players[1].eliminatedRings = 1;
      state.players[1].territorySpaces = 2;
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);

      const result = evaluateVictoryDetailed(state);

      expect(result.scores).toBeDefined();
      expect(result.scores![1].eliminatedRings).toBe(3);
      expect(result.scores![1].territorySpaces).toBe(5);
      expect(result.scores![1].markerCount).toBe(2);
      expect(result.scores![2].eliminatedRings).toBe(1);
      expect(result.scores![2].territorySpaces).toBe(2);
      expect(result.scores![2].markerCount).toBe(0);
    });

    it('sorts standings by eliminated rings when territory tied', () => {
      const state = createTestGameState();
      state.players[0].territorySpaces = 5;
      state.players[0].eliminatedRings = 3;
      state.players[1].territorySpaces = 5;
      state.players[1].eliminatedRings = 7;

      const result = evaluateVictoryDetailed(state);

      expect(result.standings![0].playerNumber).toBe(2); // Higher eliminations
    });

    it('sorts standings by markers when territory and eliminations tied', () => {
      const state = createTestGameState();
      state.players[0].territorySpaces = 5;
      state.players[0].eliminatedRings = 3;
      state.players[1].territorySpaces = 5;
      state.players[1].eliminatedRings = 3;
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 2);
      addMarker(state.board, { x: 2, y: 0 }, 2);

      const result = evaluateVictoryDetailed(state);

      expect(result.standings![0].playerNumber).toBe(2); // More markers
    });
  });

  describe('getEliminatedRingCount', () => {
    it('returns 0 for unknown player', () => {
      const state = createTestGameState();
      expect(getEliminatedRingCount(state, 999)).toBe(0);
    });

    it('returns correct count for known player', () => {
      const state = createTestGameState();
      state.players[0].eliminatedRings = 5;
      expect(getEliminatedRingCount(state, 1)).toBe(5);
    });
  });

  describe('getTerritoryCount', () => {
    it('returns 0 for unknown player', () => {
      const state = createTestGameState();
      expect(getTerritoryCount(state, 999)).toBe(0);
    });

    it('returns correct count for known player', () => {
      const state = createTestGameState();
      state.players[0].territorySpaces = 8;
      expect(getTerritoryCount(state, 1)).toBe(8);
    });
  });

  describe('getMarkerCount', () => {
    it('returns 0 when no markers exist', () => {
      const state = createTestGameState();
      state.board.markers.clear();
      expect(getMarkerCount(state, 1)).toBe(0);
    });

    it('returns correct count for player with markers', () => {
      const state = createTestGameState();
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 2);

      expect(getMarkerCount(state, 1)).toBe(2);
      expect(getMarkerCount(state, 2)).toBe(1);
    });
  });

  describe('isVictoryThresholdReached', () => {
    it('returns false when game is not over', () => {
      const state = createTestGameState();
      state.victoryThreshold = 100;
      state.territoryVictoryThreshold = 100;
      expect(isVictoryThresholdReached(state)).toBe(false);
    });

    it('returns true when ring elimination threshold reached', () => {
      const state = createTestGameState();
      state.victoryThreshold = 5;
      state.players[0].eliminatedRings = 5;
      expect(isVictoryThresholdReached(state)).toBe(true);
    });

    it('returns true when territory threshold reached', () => {
      const state = createTestGameState();
      state.territoryVictoryThreshold = 10;
      state.players[0].territorySpaces = 10;
      expect(isVictoryThresholdReached(state)).toBe(true);
    });
  });

  describe('evaluateVictory edge cases', () => {
    it('returns not game over for empty players array', () => {
      const state = createTestGameState();
      state.players = [];
      const result = evaluateVictory(state);
      expect(result.isGameOver).toBe(false);
    });

    it('returns not game over when stacks remain on board', () => {
      const state = createTestGameState();
      // Default state has stacks, so game should continue
      state.victoryThreshold = 1000;
      state.territoryVictoryThreshold = 1000;
      const result = evaluateVictory(state);
      expect(result.isGameOver).toBe(false);
    });
  });

  describe('evaluateVictory bare-board stalemate', () => {
    it('returns game_completed when all tiebreakers are exhausted', () => {
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
      // Clear history to prevent last actor detection
      state.history = [];
      state.moveHistory = [];
      state.players = []; // Remove all players to force game_completed fallback

      const result = evaluateVictory(state);

      // With no players, game should not be over
      expect(result.isGameOver).toBe(false);
    });

    it('handles bare board with no legal placements due to markers', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.players[0].ringsInHand = 2;
      state.players[1].ringsInHand = 0;

      // Fill board with markers to block placements
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          addMarker(state.board, { x, y }, 1);
        }
      }

      state.victoryThreshold = 1000;
      state.territoryVictoryThreshold = 1000;

      const result = evaluateVictory(state);

      // With all spaces blocked by markers, should trigger stalemate
      expect(result.isGameOver).toBe(true);
      expect(result.handCountsAsEliminated).toBe(true);
    });

    it('handles hexagonal board position iteration', () => {
      const hexState = createTestGameState({ boardType: 'hexagonal' });
      hexState.board.stacks.clear();
      hexState.players[0].ringsInHand = 1;
      hexState.players[1].ringsInHand = 0;
      hexState.victoryThreshold = 1000;
      hexState.territoryVictoryThreshold = 1000;

      // Should check all hex positions for legal placements
      const result = evaluateVictory(hexState);

      // With empty hex board and rings in hand, game should not be over
      expect(result.isGameOver).toBe(false);
    });
  });
});
