/**
 * AI Player Interface and Base Class
 * Defines the contract for AI players and provides base functionality
 */

import { GameState, Move } from '../../../shared/types/game';

export interface AIConfig {
  difficulty: number; // 1-10 scale
  thinkTime?: number; // Milliseconds to wait before move (for UX)
  randomness?: number; // 0-1 scale for move variation
}

export interface AIEvaluation {
  score: number;
  move: Move | null;
  reasoning?: string; // For debugging/logging
}

/**
 * Base AI Player class
 * All AI implementations should extend this class
 */
export abstract class AIPlayer {
  protected config: AIConfig;
  protected playerNumber: number;

  constructor(playerNumber: number, config: AIConfig) {
    this.playerNumber = playerNumber;
    this.config = {
      thinkTime: 500, // Default 500ms
      randomness: 0.1, // Default 10% randomness
      ...config,
    };
  }

  /**
   * Main method to select a move for the current game state
   * Must be implemented by all AI classes
   */
  abstract selectMove(gameState: GameState): Promise<Move | null>;

  /**
   * Evaluate the current board position
   * Returns a score (positive = good for AI, negative = good for opponent)
   */
  abstract evaluatePosition(gameState: GameState): number;

  /**
   * Get the AI's difficulty level (1-10)
   */
  getDifficulty(): number {
    return this.config.difficulty;
  }

  /**
   * Get the AI's player number
   */
  getPlayerNumber(): number {
    return this.playerNumber;
  }

  /**
   * Simulate thinking time for better UX
   * Returns a promise that resolves after the configured think time
   */
  protected async simulateThinking(): Promise<void> {
    if (this.config.thinkTime && this.config.thinkTime > 0) {
      return new Promise((resolve) => {
        setTimeout(resolve, this.config.thinkTime);
      });
    }
  }

  /**
   * Add randomness to move selection
   * Returns true if should pick a random move instead of best move
   */
  protected shouldPickRandomMove(): boolean {
    return Math.random() < (this.config.randomness || 0);
  }

  /**
   * Get a random element from an array
   */
  protected getRandomElement<T>(array: T[]): T | undefined {
    if (array.length === 0) return undefined;
    return array[Math.floor(Math.random() * array.length)];
  }

  /**
   * Shuffle an array (Fisher-Yates algorithm)
   */
  protected shuffleArray<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }
}

/**
 * AI Player Types
 */
export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts',
}

/**
 * AI Difficulty Presets
 */
export const AI_DIFFICULTY_PRESETS: Record<number, Partial<AIConfig>> = {
  1: { difficulty: 1, randomness: 0.5 }, // Very Easy - 50% random
  2: { difficulty: 2, randomness: 0.3 }, // Easy - 30% random
  3: { difficulty: 3, randomness: 0.2 }, // Medium-Easy - 20% random
  4: { difficulty: 4, randomness: 0.1 }, // Medium - 10% random
  5: { difficulty: 5, randomness: 0.05 }, // Medium-Hard - 5% random
  6: { difficulty: 6, randomness: 0.02 }, // Hard - 2% random
  7: { difficulty: 7, randomness: 0.01 }, // Very Hard - 1% random
  8: { difficulty: 8, randomness: 0 }, // Expert - No randomness
  9: { difficulty: 9, randomness: 0 }, // Master - No randomness
  10: { difficulty: 10, randomness: 0 }, // Grandmaster - No randomness
};
