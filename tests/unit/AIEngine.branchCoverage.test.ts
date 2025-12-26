/**
 * AIEngine.branchCoverage.test.ts
 *
 * Branch coverage tests for AIEngine.ts targeting uncovered branches:
 * - createAI and createAIFromProfile variations
 * - AI type selection and mapping
 * - Configuration management (get/remove)
 * - Diagnostics tracking
 * - Difficulty presets
 */

import {
  AIEngine,
  AIType,
  AIConfig,
  AI_DIFFICULTY_PRESETS,
} from '../../src/server/game/ai/AIEngine';
import { AIProfile, AITacticType } from '../../src/shared/types/game';

// Mock logger to prevent console output
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('AIEngine branch coverage', () => {
  let aiEngine: AIEngine;

  beforeEach(() => {
    aiEngine = new AIEngine();
  });

  describe('AI_DIFFICULTY_PRESETS', () => {
    it('has presets for all difficulty levels 1-10', () => {
      for (let i = 1; i <= 10; i++) {
        expect(AI_DIFFICULTY_PRESETS[i]).toBeDefined();
        expect(AI_DIFFICULTY_PRESETS[i].profileId).toBeDefined();
      }
    });

    it('has correct AI types for each difficulty', () => {
      expect(AI_DIFFICULTY_PRESETS[1].aiType).toBe(AIType.RANDOM);
      expect(AI_DIFFICULTY_PRESETS[2].aiType).toBe(AIType.HEURISTIC);
      expect(AI_DIFFICULTY_PRESETS[3].aiType).toBe(AIType.MINIMAX);
      expect(AI_DIFFICULTY_PRESETS[7].aiType).toBe(AIType.MCTS);
      expect(AI_DIFFICULTY_PRESETS[9].aiType).toBe(AIType.GUMBEL_MCTS);
    });

    it('has increasing think times for higher difficulties', () => {
      const thinkTimes = Object.values(AI_DIFFICULTY_PRESETS).map((p) => p.thinkTime || 0);
      for (let i = 1; i < thinkTimes.length; i++) {
        expect(thinkTimes[i]).toBeGreaterThanOrEqual(thinkTimes[i - 1]);
      }
    });

    it('has decreasing randomness for higher difficulties', () => {
      const randomness = Object.values(AI_DIFFICULTY_PRESETS).map((p) => p.randomness ?? 0);
      for (let i = 1; i < randomness.length; i++) {
        expect(randomness[i]).toBeLessThanOrEqual(randomness[i - 1]);
      }
    });
  });

  describe('createAI', () => {
    it('creates AI with default difficulty 5', () => {
      aiEngine.createAI(1);
      const config = aiEngine.getAIConfig(1);
      expect(config).toBeDefined();
      expect(config?.difficulty).toBe(5);
    });

    it('creates AI with specified difficulty', () => {
      aiEngine.createAI(1, 3);
      const config = aiEngine.getAIConfig(1);
      expect(config?.difficulty).toBe(3);
    });

    it('creates AI with specified type', () => {
      aiEngine.createAI(1, 5, AIType.MCTS);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.MCTS);
    });

    it('creates AI for different player numbers', () => {
      aiEngine.createAI(1, 3);
      aiEngine.createAI(2, 7);
      aiEngine.createAI(3, 1);

      expect(aiEngine.getAIConfig(1)?.difficulty).toBe(3);
      expect(aiEngine.getAIConfig(2)?.difficulty).toBe(7);
      expect(aiEngine.getAIConfig(3)?.difficulty).toBe(1);
    });

    it('overwrites existing AI config for same player', () => {
      aiEngine.createAI(1, 3);
      aiEngine.createAI(1, 8);
      const config = aiEngine.getAIConfig(1);
      expect(config?.difficulty).toBe(8);
    });
  });

  describe('createAIFromProfile', () => {
    it('creates AI from basic profile', () => {
      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config).toBeDefined();
      expect(config?.difficulty).toBe(5);
      expect(config?.mode).toBe('service');
    });

    it('creates AI with local mode', () => {
      const profile: AIProfile = { difficulty: 5, mode: 'local_heuristic' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.mode).toBe('local_heuristic');
    });

    it('uses preset AI type when not specified', () => {
      const profile: AIProfile = { difficulty: 1, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.RANDOM);
    });

    it('uses specified AI type from profile', () => {
      const profile: AIProfile = {
        difficulty: 1,
        mode: 'service',
        aiType: 'mcts' as AITacticType,
      };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.MCTS);
    });

    it('applies preset randomness', () => {
      const profile: AIProfile = { difficulty: 1, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.randomness).toBe(0.5);
    });

    it('applies preset thinkTime', () => {
      const profile: AIProfile = { difficulty: 1, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.thinkTime).toBe(150);
    });

    it('throws for difficulty below 1', () => {
      const profile: AIProfile = { difficulty: 0, mode: 'service' };
      expect(() => aiEngine.createAIFromProfile(1, profile)).toThrow(
        'AI difficulty must be between 1 and 10'
      );
    });

    it('throws for difficulty above 10', () => {
      const profile: AIProfile = { difficulty: 11, mode: 'service' };
      expect(() => aiEngine.createAIFromProfile(1, profile)).toThrow(
        'AI difficulty must be between 1 and 10'
      );
    });

    it('defaults mode to service when not specified', () => {
      const profile: AIProfile = { difficulty: 5 } as AIProfile;
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.mode).toBe('service');
    });

    it('creates AI for all difficulty levels', () => {
      for (let difficulty = 1; difficulty <= 10; difficulty++) {
        const profile: AIProfile = { difficulty, mode: 'service' };
        aiEngine.createAIFromProfile(difficulty, profile);
        const config = aiEngine.getAIConfig(difficulty);
        expect(config?.difficulty).toBe(difficulty);
      }
    });
  });

  describe('getAIConfig', () => {
    it('returns undefined for non-existent player', () => {
      expect(aiEngine.getAIConfig(99)).toBeUndefined();
    });

    it('returns config for existing player', () => {
      aiEngine.createAI(1, 5);
      const config = aiEngine.getAIConfig(1);
      expect(config).toBeDefined();
    });
  });

  describe('removeAI', () => {
    it('returns false for non-existent player', () => {
      expect(aiEngine.removeAI(99)).toBe(false);
    });

    it('returns true and removes existing player', () => {
      aiEngine.createAI(1, 5);
      expect(aiEngine.removeAI(1)).toBe(true);
      expect(aiEngine.getAIConfig(1)).toBeUndefined();
    });

    it('can remove and re-add AI', () => {
      aiEngine.createAI(1, 5);
      aiEngine.removeAI(1);
      aiEngine.createAI(1, 3);
      expect(aiEngine.getAIConfig(1)?.difficulty).toBe(3);
    });
  });

  describe('getDiagnostics', () => {
    it('returns undefined for non-existent player', () => {
      expect(aiEngine.getDiagnostics(99)).toBeUndefined();
    });
  });

  describe('AIType enum values', () => {
    it('has correct string values', () => {
      expect(AIType.RANDOM).toBe('random');
      expect(AIType.HEURISTIC).toBe('heuristic');
      expect(AIType.MINIMAX).toBe('minimax');
      expect(AIType.MCTS).toBe('mcts');
      expect(AIType.DESCENT).toBe('descent');
    });
  });

  describe('configuration edge cases', () => {
    it('handles multiple player configurations', () => {
      // Create AI for players 1-4
      for (let i = 1; i <= 4; i++) {
        aiEngine.createAI(i, i + 2);
      }

      // Verify all configurations
      for (let i = 1; i <= 4; i++) {
        const config = aiEngine.getAIConfig(i);
        expect(config?.difficulty).toBe(i + 2);
      }
    });

    it('handles boundary difficulty values', () => {
      aiEngine.createAI(1, 1);
      aiEngine.createAI(2, 10);

      expect(aiEngine.getAIConfig(1)?.difficulty).toBe(1);
      expect(aiEngine.getAIConfig(2)?.difficulty).toBe(10);
    });

    it('handles different AI types explicitly', () => {
      aiEngine.createAI(1, 5, AIType.RANDOM);
      aiEngine.createAI(2, 5, AIType.HEURISTIC);
      aiEngine.createAI(3, 5, AIType.MINIMAX);
      aiEngine.createAI(4, 5, AIType.MCTS);

      expect(aiEngine.getAIConfig(1)?.aiType).toBe(AIType.RANDOM);
      expect(aiEngine.getAIConfig(2)?.aiType).toBe(AIType.HEURISTIC);
      expect(aiEngine.getAIConfig(3)?.aiType).toBe(AIType.MINIMAX);
      expect(aiEngine.getAIConfig(4)?.aiType).toBe(AIType.MCTS);
    });
  });

  describe('profile AI type mapping', () => {
    it('maps random tactic type', () => {
      const profile: AIProfile = {
        difficulty: 5,
        mode: 'service',
        aiType: 'random' as AITacticType,
      };
      aiEngine.createAIFromProfile(1, profile);
      expect(aiEngine.getAIConfig(1)?.aiType).toBe(AIType.RANDOM);
    });

    it('maps heuristic tactic type', () => {
      const profile: AIProfile = {
        difficulty: 5,
        mode: 'service',
        aiType: 'heuristic' as AITacticType,
      };
      aiEngine.createAIFromProfile(1, profile);
      expect(aiEngine.getAIConfig(1)?.aiType).toBe(AIType.HEURISTIC);
    });

    it('maps minimax tactic type', () => {
      const profile: AIProfile = {
        difficulty: 5,
        mode: 'service',
        aiType: 'minimax' as AITacticType,
      };
      aiEngine.createAIFromProfile(1, profile);
      expect(aiEngine.getAIConfig(1)?.aiType).toBe(AIType.MINIMAX);
    });

    it('maps mcts tactic type', () => {
      const profile: AIProfile = {
        difficulty: 5,
        mode: 'service',
        aiType: 'mcts' as AITacticType,
      };
      aiEngine.createAIFromProfile(1, profile);
      expect(aiEngine.getAIConfig(1)?.aiType).toBe(AIType.MCTS);
    });

    it('maps descent tactic type', () => {
      const profile: AIProfile = {
        difficulty: 5,
        mode: 'service',
        aiType: 'descent' as AITacticType,
      };
      aiEngine.createAIFromProfile(1, profile);
      expect(aiEngine.getAIConfig(1)?.aiType).toBe(AIType.DESCENT);
    });
  });

  describe('preset coverage', () => {
    it('uses preset for difficulty 1 (random)', () => {
      const profile: AIProfile = { difficulty: 1, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.RANDOM);
      expect(config?.randomness).toBe(0.5);
    });

    it('uses preset for difficulty 2 (heuristic)', () => {
      const profile: AIProfile = { difficulty: 2, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.HEURISTIC);
      expect(config?.randomness).toBe(0.3);
    });

    it('uses preset for difficulty 7 (mcts)', () => {
      const profile: AIProfile = { difficulty: 7, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.MCTS);
      expect(config?.randomness).toBe(0);
    });

    it('uses preset for difficulty 10 (gumbel_mcts)', () => {
      const profile: AIProfile = { difficulty: 10, mode: 'service' };
      aiEngine.createAIFromProfile(1, profile);
      const config = aiEngine.getAIConfig(1);
      expect(config?.aiType).toBe(AIType.GUMBEL_MCTS);
      expect(config?.thinkTime).toBe(16000);
    });
  });
});
