import {
  toHUDViewModel,
  toVictoryViewModel,
  HUDViewModel,
  VictoryViewModel,
} from '../../../src/client/adapters/gameViewModels';
import { GameState, GameResult, Player, BoardState } from '../../../src/shared/types/game';
import { GameEndExplanation } from '../../../src/shared/engine/gameEndExplanation';

// Helper to create minimal game state
function createTestGameState(players: Player[]): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 0, 2: 0 },
    size: 8,
    type: 'square8',
  };

  return {
    id: 'test-game',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 36,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };
}

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

describe('gameViewModels', () => {
  describe('toHUDViewModel - gameEndExplanation weirdState derivation', () => {
    it('derives structural-stalemate weirdState from explanation', () => {
      const players = createPlayers();
      const gameState = createTestGameState(players);
      const explanation: GameEndExplanation = {
        outcomeType: 'structural_stalemate',
        victoryReasonCode: 'victory_structural_stalemate_tiebreak',
        primaryConceptId: 'structural_stalemate',
        uxCopy: {
          shortSummaryKey: 'game_end.structural_stalemate.short',
          detailedSummaryKey: 'game_end.structural_stalemate.detailed',
        },
        weirdStateContext: {
          reasonCodes: ['STRUCTURAL_STALEMATE_TIEBREAK'],
          rulesContextTags: ['structural_stalemate'],
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toHUDViewModel(gameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        gameEndExplanation: explanation,
      });

      expect(vm.weirdState).toBeDefined();
      expect(vm.weirdState?.type).toBe('structural-stalemate');
      expect(vm.weirdState?.tone).toBe('critical');
    });

    it('derives forced-elimination weirdState from LPS with ANM/FE explanation', () => {
      const players = createPlayers();
      const gameState = createTestGameState(players);
      const explanation: GameEndExplanation = {
        outcomeType: 'last_player_standing',
        victoryReasonCode: 'victory_last_player_standing',
        primaryConceptId: 'lps_real_actions',
        uxCopy: {
          shortSummaryKey: 'game_end.lps.with_anm_fe.short',
          detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
        },
        weirdStateContext: {
          reasonCodes: ['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'],
          rulesContextTags: ['anm_forced_elimination'],
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toHUDViewModel(gameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        gameEndExplanation: explanation,
      });

      expect(vm.weirdState).toBeDefined();
      expect(vm.weirdState?.type).toBe('forced-elimination');
      expect(vm.weirdState?.tone).toBe('warning');
      expect(vm.weirdState?.title).toContain('Last Player Standing');
    });

    it('prefers explanation-based weirdState over legacy detection', () => {
      const players = createPlayers();
      const gameState = createTestGameState(players);
      // Legacy detection would normally return nothing for this clean state

      const explanation: GameEndExplanation = {
        outcomeType: 'structural_stalemate',
        victoryReasonCode: 'victory_structural_stalemate_tiebreak',
        primaryConceptId: 'structural_stalemate',
        uxCopy: {
          shortSummaryKey: 'game_end.structural_stalemate.short',
          detailedSummaryKey: 'game_end.structural_stalemate.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toHUDViewModel(gameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        gameEndExplanation: explanation,
      });

      expect(vm.weirdState).toBeDefined();
      expect(vm.weirdState?.type).toBe('structural-stalemate');
    });

    it('falls back to legacy detection when explanation key is unrecognized', () => {
      const players = createPlayers();
      const gameState = createTestGameState(players);
      const explanation: GameEndExplanation = {
        outcomeType: 'ring_elimination',
        victoryReasonCode: 'victory_ring_majority',
        primaryConceptId: 'ring_majority',
        uxCopy: {
          shortSummaryKey: 'game_end.ring_elimination.short',
          detailedSummaryKey: 'game_end.ring_elimination.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toHUDViewModel(gameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        gameEndExplanation: explanation,
      });

      // Should be undefined as ring elimination is not a weird state
      expect(vm.weirdState).toBeUndefined();
    });

    it('falls back to legacy detection when explanation is null', () => {
      const players = createPlayers();
      const gameState = createTestGameState(players);

      const vm = toHUDViewModel(gameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        gameEndExplanation: null,
      });

      expect(vm.weirdState).toBeUndefined();
    });

    it('handles empty uxCopy gracefully', () => {
      const players = createPlayers();
      const gameState = createTestGameState(players);
      const explanation: GameEndExplanation = {
        outcomeType: 'structural_stalemate',
        victoryReasonCode: 'victory_structural_stalemate_tiebreak',
        primaryConceptId: 'structural_stalemate',
        uxCopy: {
          shortSummaryKey: '',
          detailedSummaryKey: '',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toHUDViewModel(gameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        gameEndExplanation: explanation,
      });

      expect(vm.weirdState).toBeUndefined();
    });
  });

  describe('toVictoryViewModel - gameEndExplanation copy variants', () => {
    const players = createPlayers();
    const gameState = createTestGameState(players);
    const gameResult: GameResult = {
      winner: 1,
      reason: 'last_player_standing',
      finalScore: {
        ringsEliminated: { 1: 0, 2: 0 },
        territorySpaces: { 1: 0, 2: 0 },
        ringsRemaining: { 1: 18, 2: 18 },
      },
    };

    it('uses LPS-specific copy when explanation has LPS key', () => {
      const explanation: GameEndExplanation = {
        outcomeType: 'last_player_standing',
        victoryReasonCode: 'victory_last_player_standing',
        primaryConceptId: 'lps_real_actions',
        uxCopy: {
          shortSummaryKey: 'game_end.lps.with_anm_fe.short',
          detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toVictoryViewModel(gameResult, players, gameState, {
        currentUserId: 'p2', // Loser perspective
        gameEndExplanation: explanation,
      });

      expect(vm?.title).toBe('👑 Last Player Standing');
      expect(vm?.description).toContain('Alice was the only player able to make real moves');
    });

    it('uses structural stalemate copy when explanation has stalemate key', () => {
      const stalemateResult: GameResult = {
        ...gameResult,
        reason: 'game_completed',
        winner: undefined,
      };
      const explanation: GameEndExplanation = {
        outcomeType: 'structural_stalemate',
        victoryReasonCode: 'victory_structural_stalemate_tiebreak',
        primaryConceptId: 'structural_stalemate',
        uxCopy: {
          shortSummaryKey: 'game_end.structural_stalemate.short',
          detailedSummaryKey: 'game_end.structural_stalemate.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toVictoryViewModel(stalemateResult, players, gameState, {
        currentUserId: 'p1',
        gameEndExplanation: explanation,
      });

      expect(vm?.title).toBe('🧱 Structural Stalemate');
      expect(vm?.description).toContain('The game reached a structural stalemate');
    });

    it('uses territory mini-region copy when explanation has mini-region key', () => {
      const territoryResult: GameResult = { ...gameResult, reason: 'territory_control' };
      const explanation: GameEndExplanation = {
        outcomeType: 'territory_control',
        victoryReasonCode: 'victory_territory_majority',
        primaryConceptId: 'territory_mini_regions',
        uxCopy: {
          shortSummaryKey: 'game_end.territory_mini_region.short',
          detailedSummaryKey: 'game_end.territory_mini_region.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toVictoryViewModel(territoryResult, players, gameState, {
        currentUserId: 'p1',
        gameEndExplanation: explanation,
      });

      expect(vm?.title).toBe('🏰 Alice Wins!');
      expect(vm?.description).toContain(
        'Victory by Territory Control after resolving the final disconnected mini-region'
      );
    });

    it('falls back to legacy copy for unrecognized key', () => {
      const explanation: GameEndExplanation = {
        outcomeType: 'ring_elimination',
        victoryReasonCode: 'victory_ring_majority',
        primaryConceptId: 'ring_majority',
        uxCopy: {
          shortSummaryKey: 'unknown.key',
          detailedSummaryKey: 'unknown.key',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toVictoryViewModel(gameResult, players, gameState, {
        currentUserId: 'p1',
        gameEndExplanation: explanation,
      });

      // Should fall back to LPS copy based on gameResult.reason
      expect(vm?.title).toBe('👑 Last Player Standing');
    });

    it('falls back to legacy copy when explanation is null', () => {
      const vm = toVictoryViewModel(gameResult, players, gameState, {
        currentUserId: 'p1',
        gameEndExplanation: null,
      });

      expect(vm?.title).toBe('👑 Last Player Standing');
    });

    it('uses "You" wording for LPS when current user is winner', () => {
      const explanation: GameEndExplanation = {
        outcomeType: 'last_player_standing',
        victoryReasonCode: 'victory_last_player_standing',
        primaryConceptId: 'lps_real_actions',
        uxCopy: {
          shortSummaryKey: 'game_end.lps.with_anm_fe.short',
          detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toVictoryViewModel(gameResult, players, gameState, {
        currentUserId: 'p1', // Winner perspective
        gameEndExplanation: explanation,
      });

      expect(vm?.description).toContain('You were the only player able to make real moves');
    });

    it('prefers detailedSummaryKey over shortSummaryKey when available', () => {
      const explanation: GameEndExplanation = {
        outcomeType: 'last_player_standing',
        victoryReasonCode: 'victory_last_player_standing',
        primaryConceptId: 'lps_real_actions',
        uxCopy: {
          shortSummaryKey: 'game_end.lps.short',
          detailedSummaryKey: 'game_end.lps.detailed',
        },
        boardType: 'square8',
        numPlayers: 2,
        winnerPlayerId: 'p1',
      };

      const vm = toVictoryViewModel(gameResult, players, gameState, {
        currentUserId: 'p1',
        gameEndExplanation: explanation,
      });

      // Should match LPS logic because detailed key starts with game_end.lps
      expect(vm?.title).toBe('👑 Last Player Standing');
    });
  });
});
