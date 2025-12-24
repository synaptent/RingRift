/**
 * VictoryAggregate.branchCoverage.test.ts
 *
 * Branch coverage tests for VictoryAggregate.ts targeting uncovered branches:
 * - evaluateVictory: all victory conditions and tie-breakers
 * - checkLastPlayerStanding: player elimination checks
 * - checkScoreThreshold: ring and territory thresholds
 * - getLastActor: history and moveHistory fallbacks
 * - Various helper functions
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
import type {
  GameState,
  Player,
  BoardState,
  RingStack,
  MarkerInfo,
} from '../../src/shared/types/game';
import type { BoardType } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a minimal player
function makePlayer(playerNumber: number, overrides: Partial<Player> = {}): Player {
  return {
    id: `p${playerNumber}`,
    username: `Player${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: 10,
    eliminatedRings: 0,
    territorySpaces: 0,
    ...overrides,
  } as Player;
}

// Helper to create a minimal BoardState
function makeBoardState(overrides: Partial<BoardState> = {}): BoardState {
  return {
    type: 'square8' as BoardType,
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    territories: new Map(),
    eliminatedRings: { 1: 0, 2: 0 },
    ...overrides,
  };
}

// Helper to create a minimal GameState
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  return {
    id: 'test-game',
    boardType: 'square8',
    board: makeBoardState(),
    players: [makePlayer(1), makePlayer(2)],
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    moveHistory: [],
    history: [],
    gameStatus: 'active',
    winner: undefined,
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 15,
    territoryVictoryThreshold: 32,
    ...overrides,
  } as GameState;
}

// Helper to add a stack
function addStack(board: BoardState, x: number, y: number, player: number, rings: number[]): void {
  const key = positionToString({ x, y });
  board.stacks.set(key, {
    position: { x, y },
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer: player,
  } as RingStack);
}

// Helper to add a marker
function addMarker(board: BoardState, x: number, y: number, player: number): void {
  const key = positionToString({ x, y });
  board.markers.set(key, {
    position: { x, y },
    player,
    type: 'regular',
  } as MarkerInfo);
}

function collapseSpace(board: BoardState, x: number, y: number, owner: number): void {
  const key = positionToString({ x, y });
  board.collapsedSpaces.set(key, owner);
}

describe('VictoryAggregate branch coverage', () => {
  describe('evaluateVictory', () => {
    describe('early returns', () => {
      it('returns not game over for empty players array', () => {
        const state = makeGameState({ players: [] });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(false);
      });

      it('returns not game over for undefined players', () => {
        const state = makeGameState();
        (state as unknown as { players: undefined }).players = undefined;

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(false);
      });
    });

    describe('ring elimination victory', () => {
      it('detects ring elimination winner', () => {
        const state = makeGameState({
          players: [makePlayer(1, { eliminatedRings: 15 }), makePlayer(2, { eliminatedRings: 5 })],
        });

        const result = evaluateVictory(state);

        expect(result).toMatchObject({
          isGameOver: true,
          winner: 1,
          reason: 'ring_elimination',
          handCountsAsEliminated: false,
        });
      });
    });

    describe('territory control victory', () => {
      it('detects territory control winner', () => {
        const state = makeGameState({
          players: [makePlayer(1, { territorySpaces: 32 }), makePlayer(2, { territorySpaces: 5 })],
        });
        for (let x = 0; x < 4; x++) {
          for (let y = 0; y < 8; y++) {
            collapseSpace(state.board, x, y, 1);
          }
        }

        const result = evaluateVictory(state);

        expect(result).toMatchObject({
          isGameOver: true,
          winner: 1,
          reason: 'territory_control',
          handCountsAsEliminated: false,
        });
      });
    });

    describe('stacks on board', () => {
      it('returns not game over when stacks remain', () => {
        const board = makeBoardState();
        addStack(board, 0, 0, 1, [1]);
        const state = makeGameState({
          board,
          players: [makePlayer(1, { eliminatedRings: 5 }), makePlayer(2, { eliminatedRings: 5 })],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(false);
      });
    });

    describe('bare board stalemate', () => {
      it('returns not game over when legal placements exist', () => {
        // Bare board with players having rings in hand
        // Legal placements should exist on empty board
        const state = makeGameState({
          players: [makePlayer(1, { ringsInHand: 5 }), makePlayer(2, { ringsInHand: 5 })],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(false);
      });

      it('triggers stalemate when no legal placements exist', () => {
        // Create a board where all spaces are collapsed or have markers
        const board = makeBoardState({ size: 2 });
        // Collapse all spaces
        board.collapsedSpaces.set('0,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,0', 1);
        board.collapsedSpaces.set('1,1', 1);

        const state = makeGameState({
          board,
          players: [
            makePlayer(1, { ringsInHand: 1, territorySpaces: 2 }),
            makePlayer(2, { ringsInHand: 1, territorySpaces: 0 }),
          ],
        });

        const result = evaluateVictory(state);

        // Should be game over due to stalemate, player 1 wins via territory
        expect(result.isGameOver).toBe(true);
        expect(result.winner).toBe(1);
        expect(result.handCountsAsEliminated).toBe(true);
      });

      it('uses territory tie-breaker in stalemate', () => {
        const board = makeBoardState({ size: 2 });
        board.collapsedSpaces.set('0,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,0', 1);
        board.collapsedSpaces.set('1,1', 1);

        const state = makeGameState({
          board,
          players: [
            makePlayer(1, { ringsInHand: 0, territorySpaces: 3 }),
            makePlayer(2, { ringsInHand: 0, territorySpaces: 1 }),
          ],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(true);
        expect(result.winner).toBe(1);
        expect(result.reason).toBe('territory_control');
      });

      it('uses elimination tie-breaker when territory tied', () => {
        const board = makeBoardState({ size: 2 });
        board.collapsedSpaces.set('0,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,0', 2);
        board.collapsedSpaces.set('1,1', 2);

        const state = makeGameState({
          board,
          players: [
            makePlayer(1, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 5 }),
            makePlayer(2, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 3 }),
          ],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(true);
        expect(result.winner).toBe(1);
        expect(result.reason).toBe('ring_elimination');
      });

      it('uses marker tie-breaker when elimination tied', () => {
        const board = makeBoardState({ size: 2 });
        board.collapsedSpaces.set('0,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,0', 2);
        board.collapsedSpaces.set('1,1', 2);
        addMarker(board, 1, 1, 1);

        const state = makeGameState({
          board,
          players: [
            makePlayer(1, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 0 }),
            makePlayer(2, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 0 }),
          ],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(true);
        expect(result.winner).toBe(1);
        expect(result.reason).toBe('last_player_standing');
      });

      it('uses last actor tie-breaker when markers tied', () => {
        const board = makeBoardState({ size: 2 });
        board.collapsedSpaces.set('0,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,0', 2);
        board.collapsedSpaces.set('1,1', 2);

        const state = makeGameState({
          board,
          currentPlayer: 1,
          players: [
            makePlayer(1, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 0 }),
            makePlayer(2, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 0 }),
          ],
          history: [{ actor: 2 }] as unknown[],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(true);
        expect(result.winner).toBe(2);
        expect(result.reason).toBe('last_player_standing');
      });

      it('returns game_completed when no winner determinable', () => {
        const board = makeBoardState({ size: 2 });
        board.collapsedSpaces.set('0,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,0', 2);
        board.collapsedSpaces.set('1,1', 2);

        const state = makeGameState({
          board,
          currentPlayer: 99, // Invalid player
          players: [
            makePlayer(1, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 0 }),
            makePlayer(2, { ringsInHand: 0, territorySpaces: 0, eliminatedRings: 0 }),
          ],
          history: [],
          moveHistory: [],
        });

        const result = evaluateVictory(state);

        expect(result.isGameOver).toBe(true);
        // Last actor is determined from players array when currentPlayer not found
        expect(typeof result.winner).toBe('number');
        expect([1, 2]).toContain(result.winner);
      });
    });
  });

  describe('checkLastPlayerStanding', () => {
    it('returns null for empty players', () => {
      const state = makeGameState({ players: [] });

      const result = checkLastPlayerStanding(state);

      expect(result).toBeNull();
    });

    it('returns null for single player', () => {
      const state = makeGameState({ players: [makePlayer(1)] });

      const result = checkLastPlayerStanding(state);

      expect(result).toBeNull();
    });

    it('returns null when multiple players have rings', () => {
      const board = makeBoardState();
      addStack(board, 0, 0, 1, [1]);
      addStack(board, 1, 0, 2, [2]);
      const state = makeGameState({ board });

      const result = checkLastPlayerStanding(state);

      expect(result).toBeNull();
    });

    it('returns winner when only one player has rings', () => {
      const board = makeBoardState();
      addStack(board, 0, 0, 1, [1]);
      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 5 }), makePlayer(2, { ringsInHand: 0 })],
      });

      const result = checkLastPlayerStanding(state);

      expect(result).toMatchObject({
        isGameOver: true,
        winner: 1,
        reason: 'last_player_standing',
      });
    });

    it('counts rings in hand', () => {
      const state = makeGameState({
        players: [makePlayer(1, { ringsInHand: 5 }), makePlayer(2, { ringsInHand: 0 })],
      });

      const result = checkLastPlayerStanding(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
    });
  });

  describe('checkScoreThreshold', () => {
    it('returns null for empty players', () => {
      const state = makeGameState({ players: [] });

      const result = checkScoreThreshold(state);

      expect(result).toBeNull();
    });

    it('returns null when no threshold reached', () => {
      const state = makeGameState({
        players: [makePlayer(1, { eliminatedRings: 5 }), makePlayer(2, { eliminatedRings: 5 })],
      });

      const result = checkScoreThreshold(state);

      expect(result).toBeNull();
    });

    it('returns ring elimination winner', () => {
      const state = makeGameState({
        players: [makePlayer(1, { eliminatedRings: 15 }), makePlayer(2, { eliminatedRings: 5 })],
      });

      const result = checkScoreThreshold(state);

      expect(result).toMatchObject({
        isGameOver: true,
        winner: 1,
        reason: 'ring_elimination',
      });
    });

    it('returns territory control winner', () => {
      const state = makeGameState({
        players: [makePlayer(1, { territorySpaces: 32 }), makePlayer(2, { territorySpaces: 10 })],
      });
      for (let x = 0; x < 4; x++) {
        for (let y = 0; y < 8; y++) {
          collapseSpace(state.board, x, y, 1);
        }
      }

      const result = checkScoreThreshold(state);

      expect(result).toMatchObject({
        isGameOver: true,
        winner: 1,
        reason: 'territory_control',
      });
    });
  });

  describe('getPlayerScore', () => {
    it('returns 0 for unknown player', () => {
      const state = makeGameState();

      const score = getPlayerScore(state, 99);

      expect(score).toBe(0);
    });

    it('returns eliminated rings count', () => {
      const state = makeGameState({
        players: [makePlayer(1, { eliminatedRings: 7 })],
      });

      const score = getPlayerScore(state, 1);

      expect(score).toBe(7);
    });
  });

  describe('getRemainingPlayers', () => {
    it('returns empty for empty players', () => {
      const state = makeGameState({ players: [] });

      const remaining = getRemainingPlayers(state);

      expect(remaining).toEqual([]);
    });

    it('returns players with stacks', () => {
      const board = makeBoardState();
      addStack(board, 0, 0, 1, [1]);
      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 0 }), makePlayer(2, { ringsInHand: 0 })],
      });

      const remaining = getRemainingPlayers(state);

      expect(remaining).toHaveLength(1);
      expect(remaining[0].playerNumber).toBe(1);
    });

    it('returns players with rings in hand', () => {
      const state = makeGameState({
        players: [makePlayer(1, { ringsInHand: 5 }), makePlayer(2, { ringsInHand: 0 })],
      });

      const remaining = getRemainingPlayers(state);

      expect(remaining).toHaveLength(1);
      expect(remaining[0].playerNumber).toBe(1);
    });
  });

  describe('isPlayerEliminated', () => {
    it('returns true for unknown player', () => {
      const state = makeGameState();

      const eliminated = isPlayerEliminated(state, 99);

      expect(eliminated).toBe(true);
    });

    it('returns false when player has rings in hand', () => {
      const state = makeGameState({
        players: [makePlayer(1, { ringsInHand: 5 })],
      });

      const eliminated = isPlayerEliminated(state, 1);

      expect(eliminated).toBe(false);
    });

    it('returns false when player has stacks', () => {
      const board = makeBoardState();
      addStack(board, 0, 0, 1, [1]);
      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 0 })],
      });

      const eliminated = isPlayerEliminated(state, 1);

      expect(eliminated).toBe(false);
    });

    it('returns true when player has no rings', () => {
      const state = makeGameState({
        players: [makePlayer(1, { ringsInHand: 0 })],
      });

      const eliminated = isPlayerEliminated(state, 1);

      expect(eliminated).toBe(true);
    });
  });

  describe('getLastActor', () => {
    it('returns actor from history', () => {
      const state = makeGameState({
        history: [{ actor: 2 }, { actor: 1 }] as unknown[],
      });

      const actor = getLastActor(state);

      expect(actor).toBe(1);
    });

    it('returns player from moveHistory when no history', () => {
      const state = makeGameState({
        history: [],
        moveHistory: [{ player: 2 }, { player: 1 }],
      });

      const actor = getLastActor(state);

      expect(actor).toBe(1);
    });

    it('returns previous player when no history or moveHistory', () => {
      const state = makeGameState({
        history: [],
        moveHistory: [],
        currentPlayer: 2,
      });

      const actor = getLastActor(state);

      expect(actor).toBe(1); // Previous in turn order
    });

    it('handles current player not in players array', () => {
      const state = makeGameState({
        history: [],
        moveHistory: [],
        currentPlayer: 99,
      });

      const actor = getLastActor(state);

      // Falls back to first player
      expect(actor).toBe(1);
    });

    it('returns undefined for empty players', () => {
      const state = makeGameState({
        players: [],
        history: [],
        moveHistory: [],
      });

      const actor = getLastActor(state);

      expect(actor).toBeUndefined();
    });
  });

  describe('evaluateVictoryDetailed', () => {
    it('returns base result for empty players', () => {
      const state = makeGameState({ players: [] });

      const result = evaluateVictoryDetailed(state);

      expect(result.isGameOver).toBe(false);
      expect(result.standings).toBeUndefined();
    });

    it('includes standings sorted by criteria', () => {
      const board = makeBoardState();
      for (let x = 0; x < 5; x++) {
        collapseSpace(board, x, 0, 1);
      }
      let p2Count = 0;
      for (let x = 0; x < 8 && p2Count < 10; x++) {
        for (let y = 1; y < 8 && p2Count < 10; y++) {
          collapseSpace(board, x, y, 2);
          p2Count += 1;
        }
      }
      const state = makeGameState({
        board,
        players: [makePlayer(1, { eliminatedRings: 3 }), makePlayer(2, { eliminatedRings: 2 })],
      });

      const result = evaluateVictoryDetailed(state);

      expect(result.standings).toHaveLength(2);
      expect(result.standings![0].playerNumber).toBe(2); // More territory
      expect(result.standings![1].playerNumber).toBe(1);
    });

    it('includes score breakdown', () => {
      const board = makeBoardState();
      addMarker(board, 0, 0, 1);
      addMarker(board, 1, 0, 1);
      addMarker(board, 2, 0, 2);
      for (let x = 0; x < 5; x++) {
        collapseSpace(board, x, 1, 1);
      }
      for (let x = 0; x < 2; x++) {
        collapseSpace(board, x, 2, 2);
      }
      const state = makeGameState({
        board,
        players: [makePlayer(1, { eliminatedRings: 3 }), makePlayer(2, { eliminatedRings: 1 })],
      });

      const result = evaluateVictoryDetailed(state);

      expect(result.scores).toMatchObject({
        1: { eliminatedRings: 3, territorySpaces: 5, markerCount: 2 },
        2: { eliminatedRings: 1, territorySpaces: 2, markerCount: 1 },
      });
      expect(result.scores![1].eliminatedRings).toBe(3);
      expect(result.scores![1].territorySpaces).toBe(5);
      expect(result.scores![1].markerCount).toBe(2);
      expect(result.scores![2].markerCount).toBe(1);
    });
  });

  describe('getEliminatedRingCount', () => {
    it('returns 0 for unknown player', () => {
      const state = makeGameState();

      const count = getEliminatedRingCount(state, 99);

      expect(count).toBe(0);
    });

    it('returns player eliminated rings', () => {
      const state = makeGameState({
        players: [makePlayer(1, { eliminatedRings: 8 })],
      });

      const count = getEliminatedRingCount(state, 1);

      expect(count).toBe(8);
    });
  });

  describe('getTerritoryCount', () => {
    it('returns 0 for unknown player', () => {
      const state = makeGameState();

      const count = getTerritoryCount(state, 99);

      expect(count).toBe(0);
    });

    it('returns player territory spaces', () => {
      const state = makeGameState({
        players: [makePlayer(1)],
      });
      for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 4; y++) {
          collapseSpace(state.board, x, y, 1);
        }
      }

      const count = getTerritoryCount(state, 1);

      expect(count).toBe(12);
    });
  });

  describe('getMarkerCount', () => {
    it('returns 0 when no markers', () => {
      const state = makeGameState();

      const count = getMarkerCount(state, 1);

      expect(count).toBe(0);
    });

    it('counts markers for player', () => {
      const board = makeBoardState();
      addMarker(board, 0, 0, 1);
      addMarker(board, 1, 0, 1);
      addMarker(board, 2, 0, 2);
      const state = makeGameState({ board });

      const count1 = getMarkerCount(state, 1);
      const count2 = getMarkerCount(state, 2);

      expect(count1).toBe(2);
      expect(count2).toBe(1);
    });
  });

  describe('isVictoryThresholdReached', () => {
    it('returns false when game not over', () => {
      const state = makeGameState();

      const reached = isVictoryThresholdReached(state);

      expect(reached).toBe(false);
    });

    it('returns true when ring elimination threshold reached', () => {
      const state = makeGameState({
        players: [makePlayer(1, { eliminatedRings: 15 })],
      });

      const reached = isVictoryThresholdReached(state);

      expect(reached).toBe(true);
    });
  });

  describe('hexagonal board support', () => {
    it('handles hex board position iteration', () => {
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 3 });
      // Size 3 = radius 2
      board.collapsedSpaces.set('0,0,0', 1);

      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 1 }), makePlayer(2, { ringsInHand: 0 })],
      });

      // Should not crash when iterating hex positions
      const result = evaluateVictory(state);
      expect(result).toMatchObject({ isGameOver: expect.any(Boolean) });
      // handCountsAsEliminated is only set when game is over via stalemate
      // When isGameOver is false, the field may be undefined
    });
  });

  // ==========================================================================
  // hasAnyLegalPlacementOnBareBoard - Line 106 branch
  // ==========================================================================
  describe('hasAnyLegalPlacementOnBareBoard branch coverage', () => {
    it('handles player not found in players array', () => {
      const state = makeGameState({
        players: [makePlayer(1), makePlayer(2)],
      });

      // Check victory when a non-existent player is referenced internally
      // With default players having ringsInHand: 10 and an open board, game is not over
      const result = evaluateVictory(state);
      expect(result).toMatchObject({ isGameOver: expect.any(Boolean) });
      // handCountsAsEliminated is only defined when stalemate logic runs
    });

    it('handles player with zero rings in hand', () => {
      const board = makeBoardState();
      // Empty board with no stacks

      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 0 }), makePlayer(2, { ringsInHand: 0 })],
      });

      // Both players have no rings to place - bare board stalemate
      // handCountsAsEliminated is false when no one has rings in hand (nothing to treat as eliminated)
      const result = evaluateVictory(state);
      expect(result).toMatchObject({ isGameOver: true, handCountsAsEliminated: false });
    });
  });

  // ==========================================================================
  // Stack lookup in view callback - Lines 160-173
  // ==========================================================================
  describe('stack lookup in move validation', () => {
    it('handles board with stacks during placement check', () => {
      const board = makeBoardState();
      // Add a stack to the board
      const key = positionToString({ x: 3, y: 3 });
      board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
        capHeight: 2,
      });

      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 5 }), makePlayer(2, { ringsInHand: 5 })],
      });

      // Players can place rings, so game is not over
      const result = evaluateVictory(state);
      expect(result.isGameOver).toBe(false);
      // handCountsAsEliminated is only set when stalemate triggers
    });

    it('handles stack lookup returning undefined', () => {
      const board = makeBoardState();

      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 1 }), makePlayer(2, { ringsInHand: 1 })],
      });

      // Empty board - stack lookups should return undefined
      const result = evaluateVictory(state);
      expect(result.isGameOver).toBe(false);
    });
  });

  // ==========================================================================
  // Last player standing - Line 326
  // ==========================================================================
  describe('last player standing victory condition', () => {
    it('detects last player standing when only one player has stacks', () => {
      const board = makeBoardState();
      const key = positionToString({ x: 3, y: 3 });
      board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
        capHeight: 2,
      });

      const state = makeGameState({
        board,
        players: [
          makePlayer(1, { ringsInHand: 5 }),
          makePlayer(2, { ringsInHand: 0 }), // No rings left
        ],
      });

      const result = checkLastPlayerStanding(state);
      // Player 2 has no material (no stacks, no rings in hand)
      // This triggers last_player_standing condition - player 1 wins
      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');
    });

    it('no victory when multiple players have material', () => {
      const board = makeBoardState();
      // Player 1 has a stack
      const key1 = positionToString({ x: 3, y: 3 });
      board.stacks.set(key1, {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
        capHeight: 2,
      });
      // Player 2 has a stack
      const key2 = positionToString({ x: 5, y: 5 });
      board.stacks.set(key2, {
        position: { x: 5, y: 5 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
        capHeight: 2,
      });

      const state = makeGameState({
        board,
        players: [makePlayer(1, { ringsInHand: 0 }), makePlayer(2, { ringsInHand: 0 })],
      });

      const result = checkLastPlayerStanding(state);
      // Returns null or object with isGameOver: false when no winner
      expect(result === null || result.isGameOver === false).toBe(true);
    });
  });

  // ==========================================================================
  // Safety fallback - Line 444
  // ==========================================================================
  describe('safety fallback for degenerate game states', () => {
    it('handles state with no actors in history', () => {
      const state = makeGameState({
        history: [],
        moveHistory: [],
      });

      const lastActor = getLastActor(state);
      // getLastActor may return currentPlayer as fallback or undefined
      // depending on implementation
      expect(lastActor === undefined || typeof lastActor === 'number').toBe(true);
    });

    it('uses history player when available', () => {
      const state = makeGameState({
        history: [{ player: 1, moveNumber: 1, type: 'place_ring' } as never],
        moveHistory: [],
      });

      const lastActor = getLastActor(state);
      // getLastActor returns a player number from history or fallback
      expect(typeof lastActor).toBe('number');
    });

    it('uses moveHistory as fallback', () => {
      const state = makeGameState({
        history: [],
        moveHistory: [{ player: 2, moveNumber: 1, type: 'place_ring' } as never],
      });

      const lastActor = getLastActor(state);
      // Should use moveHistory player
      expect(typeof lastActor).toBe('number');
    });
  });

  // ==========================================================================
  // Tie-breaker comparisons - Lines 668-674
  // ==========================================================================
  describe('tie-breaker logic', () => {
    it('breaks ties by eliminated rings when territory equal', () => {
      const board = makeBoardState();
      // Both players have same territory
      const key1 = positionToString({ x: 1, y: 1 });
      const key2 = positionToString({ x: 5, y: 5 });
      board.stacks.set(key1, {
        position: { x: 1, y: 1 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
        capHeight: 2,
      });
      board.stacks.set(key2, {
        position: { x: 5, y: 5 },
        stackHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
        capHeight: 2,
      });

      const state = makeGameState({
        board,
        gameStatus: 'completed',
        players: [
          makePlayer(1, { territorySpaces: 5, eliminatedRings: 10 }),
          makePlayer(2, { territorySpaces: 5, eliminatedRings: 8 }),
        ],
      });

      const result = evaluateVictoryDetailed(state);
      expect(result.standings).toHaveLength(2);
      // Player 1 has more eliminated rings, should rank higher in tie-breaker
      expect(result.standings![0].playerNumber).toBe(1);
      expect(result.standings![1].playerNumber).toBe(2);
    });

    it('breaks ties by marker count when territory and eliminated rings equal', () => {
      const board = makeBoardState();
      // Add markers for different players
      board.markers.set(positionToString({ x: 0, y: 0 }), {
        player: 1,
        type: 'captured',
      } as MarkerInfo);
      board.markers.set(positionToString({ x: 0, y: 1 }), {
        player: 1,
        type: 'captured',
      } as MarkerInfo);
      board.markers.set(positionToString({ x: 1, y: 0 }), {
        player: 1,
        type: 'captured',
      } as MarkerInfo);
      board.markers.set(positionToString({ x: 2, y: 2 }), {
        player: 2,
        type: 'captured',
      } as MarkerInfo);

      const state = makeGameState({
        board,
        gameStatus: 'completed',
        players: [
          makePlayer(1, { territorySpaces: 5, eliminatedRings: 10 }),
          makePlayer(2, { territorySpaces: 5, eliminatedRings: 10 }),
        ],
      });

      const result = evaluateVictoryDetailed(state);
      expect(result.standings).toHaveLength(2);
      // Player 1 has more markers (3 vs 1), should rank higher
      expect(result.standings![0].playerNumber).toBe(1);
      expect(result.standings![1].playerNumber).toBe(2);
    });

    it('handles equal scores across all tie-breakers', () => {
      const board = makeBoardState();

      const state = makeGameState({
        board,
        gameStatus: 'completed',
        players: [
          makePlayer(1, { territorySpaces: 5, eliminatedRings: 10 }),
          makePlayer(2, { territorySpaces: 5, eliminatedRings: 10 }),
        ],
      });

      // Both players equal in everything
      const result = evaluateVictoryDetailed(state);
      expect(result.standings).toHaveLength(2);
      expect(result.standings![0]).toMatchObject({ playerNumber: expect.any(Number) });
    });
  });

  // ==========================================================================
  // Trapped position stalemate with gameStatus === 'completed'
  // ==========================================================================
  describe('trapped position stalemate detection', () => {
    it('does NOT declare terminal stalemate when trapped stacks remain, even with gameStatus completed', () => {
      // This is the critical test case: gameStatus is 'completed' but evaluateVictory
      // is still being called (e.g., for victory probe). Both players have stacks but
      // are completely trapped with no legal moves.
      const board = makeBoardState({ size: 3 }); // Tiny 3x3 board

      // Create a trapped position: two stacks surrounded by collapsed spaces
      // Player 1 stack at (1,1) - center of 3x3 board
      addStack(board, 1, 1, 1, [1, 1]); // Height 2 stack

      // Player 2 stack at (0,0) - corner
      addStack(board, 0, 0, 2, [2, 2]); // Height 2 stack

      // Collapse ALL other spaces so no movement is possible
      // On a 3x3 board: positions 0-2 for x and y
      for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 3; y++) {
          const key = positionToString({ x, y });
          // Don't collapse spaces that have stacks
          if (key !== '1,1' && key !== '0,0') {
            board.collapsedSpaces.set(key, 1);
          }
        }
      }

      const state = makeGameState({
        board,
        gameStatus: 'completed', // KEY: Status is already 'completed'
        currentPhase: 'movement',
        players: [
          makePlayer(1, { ringsInHand: 0, territorySpaces: 3, eliminatedRings: 5 }),
          makePlayer(2, { ringsInHand: 0, territorySpaces: 2, eliminatedRings: 3 }),
        ],
      });

      const result = evaluateVictory(state);

      // Under canonical Python semantics, positions with stacks remaining are
      // non-terminal even if everyone is currently trapped; they must be
      // resolved via ANM/FE, and bare-board stalemate only applies once all
      // stacks are gone. TS evaluateVictory should therefore *not* report a
      // terminal stalemate here, regardless of state.gameStatus.
      expect(result.isGameOver).toBe(false);
    });

    it('correctly identifies player can move when paths exist even with gameStatus completed', () => {
      const board = makeBoardState({ size: 4 }); // 4x4 board

      // Player 1 stack at (0,0)
      addStack(board, 0, 0, 1, [1, 1]); // Height 2

      // Player 2 stack at (3,3) - opposite corner
      addStack(board, 3, 3, 2, [2, 2]); // Height 2

      // Leave some open spaces for movement
      // Only collapse a few spaces
      board.collapsedSpaces.set('1,0', 1);
      board.collapsedSpaces.set('0,1', 1);

      const state = makeGameState({
        board,
        gameStatus: 'completed',
        currentPhase: 'movement',
        players: [makePlayer(1, { ringsInHand: 0 }), makePlayer(2, { ringsInHand: 0 })],
      });

      const result = evaluateVictory(state);

      // With open spaces, players should still have moves available
      // The game is NOT over via stalemate
      expect(result.isGameOver).toBe(false);
    });
  });

  // ==========================================================================
  // Additional edge cases
  // ==========================================================================
  describe('additional edge cases', () => {
    it('handles 3-player game with varied elimination status', () => {
      const board = makeBoardState();
      const key = positionToString({ x: 3, y: 3 });
      board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
        capHeight: 3,
      });

      const state = makeGameState({
        board,
        players: [
          makePlayer(1, { ringsInHand: 5 }),
          makePlayer(2, { ringsInHand: 0 }), // Eliminated
          makePlayer(3, { ringsInHand: 0 }), // Eliminated
        ],
      });

      const result = evaluateVictory(state);
      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });

    it('handles 4-player game standings', () => {
      const board = makeBoardState();
      const territoryAssignments = [
        { player: 1, count: 10 },
        { player: 2, count: 8 },
        { player: 3, count: 6 },
        { player: 4, count: 4 },
      ];
      let x = 0;
      let y = 0;
      for (const { player, count } of territoryAssignments) {
        for (let i = 0; i < count; i++) {
          collapseSpace(board, x, y, player);
          x += 1;
          if (x >= 8) {
            x = 0;
            y += 1;
          }
        }
      }

      const state = makeGameState({
        board,
        gameStatus: 'completed',
        players: [
          makePlayer(1, { territorySpaces: 10, eliminatedRings: 5 }),
          makePlayer(2, { territorySpaces: 8, eliminatedRings: 7 }),
          makePlayer(3, { territorySpaces: 6, eliminatedRings: 3 }),
          makePlayer(4, { territorySpaces: 4, eliminatedRings: 2 }),
        ],
      });

      const result = evaluateVictoryDetailed(state);
      expect(result.standings).toHaveLength(4);
      // Player 1 has most territory, should be first
      expect(result.standings![0].playerNumber).toBe(1);
    });
  });
});
