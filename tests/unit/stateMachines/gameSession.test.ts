/**
 * Unit tests for gameSession.ts state machine
 * Covers deriveGameSessionStatus function and all GameSessionStatus variants
 */
import {
  deriveGameSessionStatus,
  type GameSessionStatus,
} from '../../../src/shared/stateMachines/gameSession';
import type { GameState, GameResult } from '../../../src/shared/types/game';

describe('gameSession state machine', () => {
  // Create a minimal mock GameState for testing
  const createMockGameState = (overrides: Partial<GameState> = {}): GameState => ({
    id: 'test-game-123',
    boardType: 'square8',
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    },
    players: [],
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 10, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
    ...overrides,
  });

  const mockGameResult: GameResult = {
    winner: 1,
    reason: 'ring_elimination',
    finalScore: {
      ringsEliminated: { 1: 0, 2: 18 },
      territorySpaces: { 1: 10, 2: 5 },
      ringsRemaining: { 1: 18, 2: 0 },
    },
  };

  describe('deriveGameSessionStatus', () => {
    describe('waiting status', () => {
      it('should derive waiting_for_players status from waiting gameStatus', () => {
        const state = createMockGameState({ gameStatus: 'waiting' });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('waiting_for_players');
        expect(sessionStatus.gameId).toBe('test-game-123');
        expect(sessionStatus.status).toBe('waiting');
      });

      it('should not include result for waiting status', () => {
        const state = createMockGameState({ gameStatus: 'waiting' });
        const sessionStatus = deriveGameSessionStatus(state, mockGameResult);

        expect(sessionStatus.kind).toBe('waiting_for_players');
        // Result is not included for waiting status
        expect('result' in sessionStatus).toBe(false);
      });
    });

    describe('active/paused status', () => {
      it('should derive active_turn status from active gameStatus', () => {
        const state = createMockGameState({
          gameStatus: 'active',
          currentPlayer: 2,
          currentPhase: 'movement',
        });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('active_turn');
        expect(sessionStatus.gameId).toBe('test-game-123');
        expect(sessionStatus.status).toBe('active');
        if (sessionStatus.kind === 'active_turn') {
          expect(sessionStatus.currentPlayer).toBe(2);
          expect(sessionStatus.phase).toBe('movement');
        }
      });

      it('should derive active_turn status from paused gameStatus', () => {
        const state = createMockGameState({
          gameStatus: 'paused',
          currentPlayer: 1,
          currentPhase: 'capture',
        });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('active_turn');
        expect(sessionStatus.status).toBe('paused');
        if (sessionStatus.kind === 'active_turn') {
          expect(sessionStatus.currentPlayer).toBe(1);
          expect(sessionStatus.phase).toBe('capture');
        }
      });

      it('should include all game phases correctly', () => {
        const phases = [
          'ring_placement',
          'movement',
          'capture',
          'chain_capture',
          'line_processing',
          'territory_processing',
        ] as const;

        for (const phase of phases) {
          const state = createMockGameState({
            gameStatus: 'active',
            currentPhase: phase,
          });
          const sessionStatus = deriveGameSessionStatus(state);

          expect(sessionStatus.kind).toBe('active_turn');
          if (sessionStatus.kind === 'active_turn') {
            expect(sessionStatus.phase).toBe(phase);
          }
        }
      });
    });

    describe('abandoned status', () => {
      it('should derive abandoned status from abandoned gameStatus', () => {
        const state = createMockGameState({ gameStatus: 'abandoned' });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('abandoned');
        expect(sessionStatus.gameId).toBe('test-game-123');
        expect(sessionStatus.status).toBe('abandoned');
      });

      it('should include result for abandoned status when provided', () => {
        const state = createMockGameState({ gameStatus: 'abandoned' });
        const abandonResult: GameResult = {
          ...mockGameResult,
          reason: 'abandonment',
        };
        const sessionStatus = deriveGameSessionStatus(state, abandonResult);

        expect(sessionStatus.kind).toBe('abandoned');
        if (sessionStatus.kind === 'abandoned') {
          expect(sessionStatus.result).toBe(abandonResult);
        }
      });

      it('should have undefined result for abandoned status when not provided', () => {
        const state = createMockGameState({ gameStatus: 'abandoned' });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('abandoned');
        if (sessionStatus.kind === 'abandoned') {
          expect(sessionStatus.result).toBeUndefined();
        }
      });
    });

    describe('completed/finished status', () => {
      it('should derive completed status from completed gameStatus', () => {
        const state = createMockGameState({ gameStatus: 'completed' });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('completed');
        expect(sessionStatus.gameId).toBe('test-game-123');
        expect(sessionStatus.status).toBe('completed');
      });

      it('should derive completed status from finished gameStatus (legacy)', () => {
        const state = createMockGameState({ gameStatus: 'finished' });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('completed');
        expect(sessionStatus.status).toBe('finished');
      });

      it('should include result for completed status when provided', () => {
        const state = createMockGameState({ gameStatus: 'completed' });
        const sessionStatus = deriveGameSessionStatus(state, mockGameResult);

        expect(sessionStatus.kind).toBe('completed');
        if (sessionStatus.kind === 'completed') {
          expect(sessionStatus.result).toBe(mockGameResult);
          expect(sessionStatus.result?.winner).toBe(1);
          expect(sessionStatus.result?.reason).toBe('ring_elimination');
        }
      });

      it('should have undefined result for completed status when not provided', () => {
        const state = createMockGameState({ gameStatus: 'completed' });
        const sessionStatus = deriveGameSessionStatus(state);

        expect(sessionStatus.kind).toBe('completed');
        if (sessionStatus.kind === 'completed') {
          expect(sessionStatus.result).toBeUndefined();
        }
      });
    });

    describe('gameId preservation', () => {
      it('should preserve gameId across all status types', () => {
        const statuses = [
          'waiting',
          'active',
          'paused',
          'abandoned',
          'completed',
          'finished',
        ] as const;

        for (const gameStatus of statuses) {
          const state = createMockGameState({
            id: 'unique-game-id',
            gameStatus: gameStatus as any,
          });
          const sessionStatus = deriveGameSessionStatus(state);

          expect(sessionStatus.gameId).toBe('unique-game-id');
        }
      });
    });

    describe('different result reasons', () => {
      const resultReasons = [
        'ring_elimination',
        'territory_control',
        'last_player_standing',
        'timeout',
        'resignation',
        'draw',
        'abandonment',
        'game_completed',
      ] as const;

      for (const reason of resultReasons) {
        it(`should include result with reason: ${reason}`, () => {
          const state = createMockGameState({ gameStatus: 'completed' });
          const result: GameResult = {
            ...mockGameResult,
            reason,
          };
          const sessionStatus = deriveGameSessionStatus(state, result);

          expect(sessionStatus.kind).toBe('completed');
          if (sessionStatus.kind === 'completed') {
            expect(sessionStatus.result?.reason).toBe(reason);
          }
        });
      }
    });

    describe('winner field handling', () => {
      it('should preserve winner in result when present', () => {
        const state = createMockGameState({ gameStatus: 'completed' });
        const sessionStatus = deriveGameSessionStatus(state, mockGameResult);

        if (sessionStatus.kind === 'completed') {
          expect(sessionStatus.result?.winner).toBe(1);
        }
      });

      it('should preserve undefined winner in result (draw)', () => {
        const state = createMockGameState({ gameStatus: 'completed' });
        const drawResult: GameResult = {
          reason: 'draw',
          finalScore: mockGameResult.finalScore,
        };
        const sessionStatus = deriveGameSessionStatus(state, drawResult);

        if (sessionStatus.kind === 'completed') {
          expect(sessionStatus.result?.winner).toBeUndefined();
        }
      });
    });
  });

  describe('type narrowing', () => {
    it('should allow type narrowing on WaitingForPlayersSession', () => {
      const state = createMockGameState({ gameStatus: 'waiting' });
      const sessionStatus = deriveGameSessionStatus(state);

      if (sessionStatus.kind === 'waiting_for_players') {
        // TypeScript should allow access to WaitingForPlayersSession fields
        expect(sessionStatus.status).toBe('waiting');
      }
    });

    it('should allow type narrowing on ActiveTurnSession', () => {
      const state = createMockGameState({
        gameStatus: 'active',
        currentPlayer: 2,
        currentPhase: 'chain_capture',
      });
      const sessionStatus = deriveGameSessionStatus(state);

      if (sessionStatus.kind === 'active_turn') {
        // TypeScript should allow access to ActiveTurnSession fields
        expect(sessionStatus.currentPlayer).toBe(2);
        expect(sessionStatus.phase).toBe('chain_capture');
      }
    });

    it('should allow type narrowing on CompletedSession', () => {
      const state = createMockGameState({ gameStatus: 'completed' });
      const sessionStatus = deriveGameSessionStatus(state, mockGameResult);

      if (sessionStatus.kind === 'completed') {
        // TypeScript should allow access to CompletedSession fields
        expect(sessionStatus.result?.winner).toBe(1);
      }
    });

    it('should allow type narrowing on AbandonedSession', () => {
      const state = createMockGameState({ gameStatus: 'abandoned' });
      const sessionStatus = deriveGameSessionStatus(state, mockGameResult);

      if (sessionStatus.kind === 'abandoned') {
        // TypeScript should allow access to AbandonedSession fields
        expect(sessionStatus.result?.winner).toBe(1);
      }
    });
  });
});
