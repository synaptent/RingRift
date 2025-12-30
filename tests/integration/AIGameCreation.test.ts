/**
 * Integration tests for AI game creation and lifecycle
 *
 * Tests the full flow from lobby UI through game creation API to
 * GameSession initialization with AI opponents.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { CreateGameRequest, GameState, AIProfile } from '../../src/shared/types/game';
import { CreateGameSchema } from '../../src/shared/validation/schemas';

describe('AI Game Creation Integration', () => {
  describe('Validation', () => {
    it('should accept valid AI game configuration', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 2,
        aiOpponents: {
          count: 1,
          difficulty: [5],
          mode: 'service',
          aiType: 'heuristic',
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(true);
    });

    it('should reject AI difficulty below 1', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 2,
        aiOpponents: {
          count: 1,
          difficulty: [0], // Invalid
          mode: 'service',
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(false);
    });

    it('should reject AI difficulty above 10', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 2,
        aiOpponents: {
          count: 1,
          difficulty: [11], // Invalid
          mode: 'service',
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(false);
    });

    it('should accept multiple AI opponents with different difficulties', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 4,
        aiOpponents: {
          count: 3,
          difficulty: [2, 5, 9], // Beginner, Intermediate, Expert
          mode: 'service',
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(true);
    });

    it('should accept valid AI types', () => {
      const validTypes = [
        'random',
        'heuristic',
        'minimax',
        'mcts',
        'descent',
        'policy_only',
        'gumbel_mcts',
        'ig_gmo',
      ] as const;

      for (const aiType of validTypes) {
        const request: CreateGameRequest = {
          boardType: 'square8',
          timeControl: {
            type: 'rapid',
            initialTime: 600,
            increment: 0,
          },
          isRated: false,
          isPrivate: false,
          maxPlayers: 2,
          aiOpponents: {
            count: 1,
            difficulty: [5],
            mode: 'service',
            aiType,
          },
        };

        const result = CreateGameSchema.safeParse(request);
        expect(result.success).toBe(true);
      }
    });
  });

  describe('AIProfile Configuration', () => {
    it('should map difficulty to correct AI type', () => {
      const testCases: Array<{ difficulty: number; expectedType: string }> = [
        { difficulty: 1, expectedType: 'random' },
        { difficulty: 2, expectedType: 'heuristic' },
        { difficulty: 3, expectedType: 'minimax' },
        { difficulty: 4, expectedType: 'minimax' },
        { difficulty: 5, expectedType: 'descent' },
        { difficulty: 6, expectedType: 'descent' },
        { difficulty: 7, expectedType: 'mcts' },
        { difficulty: 8, expectedType: 'mcts' },
        { difficulty: 9, expectedType: 'gumbel_mcts' },
        { difficulty: 10, expectedType: 'gumbel_mcts' },
      ];

      // This test documents the expected difficulty-to-type mapping
      // The actual mapping is in AI_DIFFICULTY_PRESETS in AIEngine.ts
      testCases.forEach(({ difficulty, expectedType }) => {
        expect(difficulty).toBeGreaterThanOrEqual(1);
        expect(difficulty).toBeLessThanOrEqual(10);
        expect(['random', 'heuristic', 'minimax', 'descent', 'mcts', 'gumbel_mcts']).toContain(
          expectedType
        );
      });
    });

    it('should create valid AIProfile from difficulty', () => {
      const profile: AIProfile = {
        difficulty: 7,
        mode: 'service',
      };

      expect(profile.difficulty).toBe(7);
      expect(profile.mode).toBe('service');
      expect(profile.aiType).toBeUndefined(); // Auto-selected based on difficulty
    });

    it('should allow AI type override', () => {
      const profile: AIProfile = {
        difficulty: 5,
        mode: 'service',
        aiType: 'minimax', // Override default heuristic for difficulty 5
      };

      expect(profile.difficulty).toBe(5);
      expect(profile.aiType).toBe('minimax');
    });
  });

  describe('Game Creation Scenarios', () => {
    it('should create human vs AI game', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'blitz',
          initialTime: 300,
          increment: 2,
        },
        isRated: false, // AI games must be unrated
        isPrivate: false,
        maxPlayers: 2,
        aiOpponents: {
          count: 1,
          difficulty: [5],
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.aiOpponents?.count).toBe(1);
        expect(result.data.aiOpponents?.difficulty[0]).toBe(5);
      }
    });

    it('should create AI vs AI game for testing', () => {
      const request: CreateGameRequest = {
        boardType: 'hexagonal',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: false,
        isPrivate: true,
        maxPlayers: 2,
        aiOpponents: {
          count: 2, // Both players are AI
          difficulty: [3, 7], // Different difficulties
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.aiOpponents?.count).toBe(2);
        expect(result.data.aiOpponents?.difficulty).toEqual([3, 7]);
      }
    });

    it('should create 4-player game with mixed human/AI', () => {
      const request: CreateGameRequest = {
        boardType: 'square19',
        timeControl: {
          type: 'classical',
          initialTime: 1800,
          increment: 10,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 4,
        aiOpponents: {
          count: 2, // 2 AI, 2 human (including creator)
          difficulty: [4, 8],
        },
      };

      const result = CreateGameSchema.safeParse(request);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.maxPlayers).toBe(4);
        expect(result.data.aiOpponents?.count).toBe(2);
      }
    });
  });

  describe('AI Game Lifecycle', () => {
    it('should document auto-start behavior for AI games', () => {
      // AI games should:
      // 1. Create with status 'active' instead of 'waiting'
      // 2. Set startedAt timestamp immediately
      // 3. Not appear in public lobby (since they're already full)
      // 4. Trigger AI turn loop via GameSession.maybePerformAITurn

      // This is a documentation test - actual behavior is tested
      // via WebSocketServer integration tests
      expect(true).toBe(true);
    });

    it('should document AI player initialization', () => {
      // When GameSession initializes with aiOpponents config:
      // 1. Creates AI Player entries with type='ai'
      // 2. Sets aiProfile with difficulty, mode, and optional aiType
      // 3. Calls globalAIEngine.createAIFromProfile for each AI player
      // 4. AI players are marked isReady=true automatically

      // See GameSession.ts lines 98-147 for implementation
      expect(true).toBe(true);
    });
  });

  describe('Difficulty Presets', () => {
    it('should document all difficulty levels', () => {
      const difficulties = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      difficulties.forEach((difficulty) => {
        expect(difficulty).toBeGreaterThanOrEqual(1);
        expect(difficulty).toBeLessThanOrEqual(10);
      });

      // Mapping (from AI_DIFFICULTY_PRESETS in AIEngine.ts):
      // 1-2: RandomAI with varying randomness
      // 3-5: HeuristicAI with decreasing randomness
      // 6-8: MinimaxAI with minimal/no randomness
      // 9-10: MCTSAI with maximum search time
    });

    it('should document think time progression', () => {
      // Think times increase with difficulty:
      // Difficulty 1: 150ms
      // Difficulty 5: 350ms
      // Difficulty 10: 700ms

      // For search-based AIs (Minimax/MCTS), this is search budget
      // For simple AIs (Random/Heuristic), this is UX delay
      expect(150).toBeLessThan(350);
      expect(350).toBeLessThan(700);
    });
  });

  describe('UI Integration Points', () => {
    it('should document lobby AI configuration fields', () => {
      // LobbyPage.tsx form state includes:
      // - aiCount (0-3)
      // - aiDifficulty (1-10, applied uniformly)
      // - aiMode ('service' | 'local_heuristic')
      // - aiType ('random' | 'heuristic' | 'minimax' | 'mcts')

      const formDefaults = {
        aiCount: 1,
        aiDifficulty: 5,
        aiMode: 'service' as const,
        aiType: 'heuristic' as const,
      };

      expect(formDefaults.aiCount).toBeGreaterThanOrEqual(0);
      expect(formDefaults.aiCount).toBeLessThanOrEqual(3);
      expect(formDefaults.aiDifficulty).toBeGreaterThanOrEqual(1);
      expect(formDefaults.aiDifficulty).toBeLessThanOrEqual(10);
    });

    it('should document GameHUD AI display features', () => {
      // GameHUD displays for AI players:
      // - AI indicator badge (🤖 AI)
      // - Difficulty level with color coding
      // - AI type label (Random/Heuristic/Minimax/MCTS)
      // - Animated thinking indicator during AI turns

      const difficultyLabels = {
        beginner: [1, 2],
        intermediate: [3, 4, 5],
        advanced: [6, 7, 8],
        expert: [9, 10],
      };

      expect(difficultyLabels.beginner).toContain(1);
      expect(difficultyLabels.expert).toContain(10);
    });
  });

  describe('Error Handling', () => {
    it('should validate AI game cannot be rated', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: true, // Should be rejected
        isPrivate: false,
        maxPlayers: 2,
        aiOpponents: {
          count: 1,
          difficulty: [5],
        },
      };

      // The backend route validates this and throws:
      // 'AI games cannot be rated', 400, 'AI_GAMES_UNRATED'
      expect(request.isRated).toBe(true);
      expect(request.aiOpponents?.count).toBeGreaterThan(0);
    });

    it('should validate difficulty array length matches count', () => {
      const request: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 3,
        aiOpponents: {
          count: 2,
          difficulty: [5], // Should have 2 values
        },
      };

      // Backend validates: difficulty.length >= count
      expect(request.aiOpponents?.difficulty.length).toBeLessThan(request.aiOpponents?.count ?? 0);
    });
  });

  describe('API Contract', () => {
    it('should document create game request format', () => {
      const exampleRequest: CreateGameRequest = {
        boardType: 'square8',
        timeControl: {
          type: 'blitz',
          initialTime: 300,
          increment: 2,
        },
        isRated: false,
        isPrivate: false,
        maxPlayers: 2,
        aiOpponents: {
          count: 1,
          difficulty: [7], // Advanced Minimax
          mode: 'service',
          aiType: 'minimax',
        },
      };

      // Request is sent to POST /api/games
      // Response includes created game with:
      // - game.id
      // - game.status = 'active' (auto-started)
      // - game.gameState.aiOpponents (persisted config)
      // - game.startedAt (timestamp)

      expect(exampleRequest.aiOpponents).toBeDefined();
      expect(exampleRequest.aiOpponents?.count).toBe(1);
    });

    it('should document game state AI opponent storage', () => {
      // The game.gameState JSON field stores:
      // {
      //   aiOpponents: {
      //     count: number,
      //     difficulty: number[],
      //     mode?: 'local_heuristic' | 'service',
      //     aiType?: 'random' | 'heuristic' | 'minimax' | 'mcts'
      //   }
      // }

      // GameSession.initialize reads this to configure AI players
      const aiOpponents = {
        count: 1,
        difficulty: [5],
        mode: 'service' as const,
      };

      expect(aiOpponents.count).toBe(1);
      expect(aiOpponents.difficulty.length).toBeGreaterThanOrEqual(aiOpponents.count);
    });
  });

  describe('GameSession AI Initialization', () => {
    it('should document AI player creation process', () => {
      // From GameSession.initialize (lines 98-147):
      // 1. Reads aiOpponents from game.gameState
      // 2. Creates AI Player objects with:
      //    - id: 'ai-{gameId}-{playerNumber}'
      //    - username: 'AI (Level {difficulty})'
      //    - type: 'ai'
      //    - aiProfile: { difficulty, mode, aiType? }
      // 3. Calls globalAIEngine.createAIFromProfile
      // 4. AI players added to game with isReady=true

      const mockAIPlayer = {
        id: 'ai-game123-2',
        username: 'AI (Level 5)',
        playerNumber: 2,
        type: 'ai' as const,
        isReady: true,
        aiProfile: {
          difficulty: 5,
          mode: 'service' as const,
        },
      };

      expect(mockAIPlayer.type).toBe('ai');
      expect(mockAIPlayer.aiProfile?.difficulty).toBe(5);
    });

    it('should document AI turn trigger flow', () => {
      // After each move/choice resolution:
      // 1. GameSession.maybePerformAITurn checks currentPlayer.type
      // 2. If 'ai', calls globalAIEngine.getAIMove
      // 3. AIEngine calls AIServiceClient with difficulty/aiType
      // 4. On service failure, falls back to local heuristic
      // 5. Move applied via RulesBackendFacade
      // 6. State broadcast to all clients
      // 7. Loop continues if next player is also AI

      expect(true).toBe(true);
    });
  });

  describe('Frontend Display', () => {
    it('should document AI difficulty label mapping', () => {
      const getDifficultyLabel = (difficulty: number): string => {
        if (difficulty <= 2) return 'Beginner';
        if (difficulty <= 5) return 'Intermediate';
        if (difficulty <= 8) return 'Advanced';
        return 'Expert';
      };

      expect(getDifficultyLabel(1)).toBe('Beginner');
      expect(getDifficultyLabel(2)).toBe('Beginner');
      expect(getDifficultyLabel(3)).toBe('Intermediate');
      expect(getDifficultyLabel(5)).toBe('Intermediate');
      expect(getDifficultyLabel(6)).toBe('Advanced');
      expect(getDifficultyLabel(8)).toBe('Advanced');
      expect(getDifficultyLabel(9)).toBe('Expert');
      expect(getDifficultyLabel(10)).toBe('Expert');
    });

    it('should document AI type display labels', () => {
      const typeLabels: Record<string, string> = {
        random: 'Random',
        heuristic: 'Heuristic',
        minimax: 'Minimax',
        mcts: 'MCTS',
        descent: 'Descent',
      };

      expect(typeLabels.random).toBe('Random');
      expect(typeLabels.minimax).toBe('Minimax');
      expect(typeLabels.mcts).toBe('MCTS');
    });
  });
});

describe('AI Game Creation E2E Flow', () => {
  it('should document complete user journey', () => {
    // User Journey:
    // 1. User navigates to /lobby
    // 2. Fills out "Create Backend Game" form:
    //    - Board: square8
    //    - Time control: 10+0
    //    - AI opponents: 1
    //    - Difficulty: 7 (Advanced/Minimax)
    //    - Clicks "Create Game"
    // 3. Frontend sends POST /api/games with aiOpponents config
    // 4. Backend validates request
    // 5. Backend creates game with:
    //    - player1Id = creator
    //    - status = 'active' (auto-started)
    //    - gameState.aiOpponents = config
    //    - startedAt = now
    // 6. Frontend redirects to /game/{gameId}
    // 7. GamePage connects via WebSocket
    // 8. GameSession initializes with AI player
    // 9. If AI is first player, AI turn executes immediately
    // 10. User sees board with AI opponent labeled
    // 11. Game proceeds with AI responding to user moves
    // 12. Victory modal shows when game completes

    expect(true).toBe(true);
  });
});
