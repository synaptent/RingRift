/**
 * AI Engine - Manages AI Players and Move Selection
 * Delegates to Python AI microservice for move generation
 */

import {
  GameState,
  Move,
  AIProfile,
  AITacticType,
  AIControlMode,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  positionToString,
} from '../../../shared/types/game';
import { getAIServiceClient, AIType as ServiceAIType } from '../../services/AIServiceClient';
import { logger } from '../../utils/logger';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts',
}

export interface AIConfig {
  difficulty: number;
  thinkTime?: number;
  randomness?: number;
  /** Tactical engine chosen for this AI config. */
  aiType?: AIType;
  /** How this AI makes decisions about moves/choices. */
  mode?: AIControlMode;
}

export const AI_DIFFICULTY_PRESETS: Record<number, Partial<AIConfig>> = {
  1: { randomness: 0.5, thinkTime: 500 },
  2: { randomness: 0.3, thinkTime: 700 },
  3: { randomness: 0.2, thinkTime: 1000 },
  4: { randomness: 0.1, thinkTime: 1200 },
  5: { randomness: 0.05, thinkTime: 1500 },
  6: { randomness: 0.02, thinkTime: 2000 },
  7: { randomness: 0.01, thinkTime: 2500 },
  8: { randomness: 0, thinkTime: 3000 },
  9: { randomness: 0, thinkTime: 4000 },
  10: { randomness: 0, thinkTime: 5000 },
};

export class AIEngine {
  private aiConfigs: Map<number, AIConfig> = new Map();

  /**
   * Create/configure an AI player
   * @param playerNumber - The player number for this AI
   * @param difficulty - Difficulty level (1-10)
   * @param type - AI type (optional, auto-selected based on difficulty if not provided)
   */
  createAI(playerNumber: number, difficulty: number = 5, type?: AIType): void {
    // Backwards-compatible wrapper around createAIFromProfile.
    const profile: AIProfile = {
      difficulty,
      mode: 'service',
      ...(type && { aiType: this.mapAITypeToTactic(type) }),
    };

    this.createAIFromProfile(playerNumber, profile);
  }

  /**
   * Configure an AI player from a rich AIProfile. This is the
   * preferred entry point for new code paths.
   */
  createAIFromProfile(playerNumber: number, profile: AIProfile): void {
    const difficulty = profile.difficulty;

    // Validate difficulty
    if (difficulty < 1 || difficulty > 10) {
      throw new Error('AI difficulty must be between 1 and 10');
    }

    const basePreset = AI_DIFFICULTY_PRESETS[difficulty] || {};
    const aiType = profile.aiType
      ? this.mapAITacticToAIType(profile.aiType)
      : this.selectAITypeForDifficulty(difficulty);

    const config: AIConfig = {
      ...basePreset,
      difficulty,
      aiType,
      mode: profile.mode ?? 'service',
    };

    this.aiConfigs.set(playerNumber, config);

    logger.info('AI player configured from profile', {
      playerNumber,
      difficulty,
      aiType,
      mode: config.mode,
    });
  }

  /**
   * Get an AI config by player number
   */
  getAIConfig(playerNumber: number): AIConfig | undefined {
    return this.aiConfigs.get(playerNumber);
  }

  /**
   * Remove an AI player
   */
  removeAI(playerNumber: number): boolean {
    return this.aiConfigs.delete(playerNumber);
  }

  /**
   * Get move from AI player via Python microservice
   * @param playerNumber - The player number
   * @param gameState - Current game state
   * @returns The selected move or null if no valid moves
   */
  async getAIMove(playerNumber: number, gameState: GameState): Promise<Move | null> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      // Call Python AI service. Prefer any explicit aiType derived from
      // AIProfile, falling back to a difficulty-based default.
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const response = await getAIServiceClient().getAIMove(
        gameState,
        playerNumber,
        config.difficulty,
        aiType as unknown as ServiceAIType
      );

      const normalizedMove = this.normalizeServiceMove(response.move, gameState, playerNumber);

      logger.info('AI move generated', {
        playerNumber,
        moveType: normalizedMove?.type,
        evaluation: response.evaluation,
        thinkingTime: response.thinking_time_ms,
        aiType: response.ai_type,
      });

      return normalizedMove;
    } catch (error) {
      logger.error('Failed to get AI move from service, falling back to local heuristic', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      // Fallback to local heuristic
      return this.getLocalAIMove(playerNumber, gameState);
    }
  }

  /**
   * Generate a move using local heuristics when the AI service is unavailable.
   * Uses RuleEngine to find valid moves and selects one randomly.
   */
  private getLocalAIMove(playerNumber: number, gameState: GameState): Move | null {
    try {
      const boardManager = new BoardManager(gameState.boardType);
      const ruleEngine = new RuleEngine(boardManager, gameState.boardType);

      let validMoves = ruleEngine.getValidMoves(gameState);

      // Enforce "must move placed stack" rule if applicable, since RuleEngine
      // doesn't track per-turn state but GameEngine does. We infer it from history.
      if (gameState.currentPhase === 'movement' || gameState.currentPhase === 'capture') {
        const lastMove = gameState.moveHistory[gameState.moveHistory.length - 1];
        // If the last move was a placement by the current player in the same turn sequence...
        // Actually, simpler: if we are in movement/capture, and the last move was 'place_ring',
        // then we must move that stack.
        if (
          lastMove &&
          lastMove.type === 'place_ring' &&
          lastMove.player === gameState.currentPlayer &&
          lastMove.to
        ) {
          const placedKey = positionToString(lastMove.to);
          validMoves = validMoves.filter((m) => {
            // Only filter movement/capture moves
            if (
              m.type === 'move_stack' ||
              m.type === 'move_ring' ||
              m.type === 'overtaking_capture' ||
              m.type === 'build_stack'
            ) {
              return m.from && positionToString(m.from) === placedKey;
            }
            return true;
          });
        }
      }

      if (validMoves.length === 0) {
        return null;
      }

      // Simple random selection for fallback
      const randomIndex = Math.floor(Math.random() * validMoves.length);
      const selectedMove = validMoves[randomIndex];

      logger.info('Local AI fallback move generated', {
        playerNumber,
        moveType: selectedMove.type,
      });

      return selectedMove;
    } catch (error) {
      logger.error('Failed to generate local AI move', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      return null;
    }
  }

  /**
   * Normalise a move returned from the Python AI service so that it
   * respects the backend placement semantics:
   * - Use the canonical 'place_ring' type for ring placements.
   * - On existing stacks, enforce exactly 1 ring per placement and set
   *   placedOnStack=true.
   * - On empty cells, allow small multi-ring placements by filling in
   *   placementCount when the service omits it, clamped by the
   *   player’s ringsInHand.
   *
   * This keeps the AI service relatively agnostic of RingRift’s
   * evolving placement rules while ensuring GameEngine/RuleEngine see
   * well-formed moves.
   */
  private normalizeServiceMove(
    move: Move | null,
    gameState: GameState,
    playerNumber: number
  ): Move | null {
    if (!move) {
      return null;
    }

    // Defensive: if board/players are missing (e.g. in unit tests that
    // mock GameState), return the move as-is.
    if (!gameState.board || !Array.isArray(gameState.players)) {
      return move;
    }

    const normalized: Move = { ...move };

    // Normalise any historical 'place' type to the canonical
    // 'place_ring'.
    if (normalized.type === ('place' as any)) {
      normalized.type = 'place_ring';
    }

    if (normalized.type !== 'place_ring') {
      return normalized;
    }

    const playerState = gameState.players.find((p) => p.playerNumber === playerNumber);
    const ringsInHand = playerState?.ringsInHand ?? 0;

    if (!normalized.to || ringsInHand <= 0) {
      // Let RuleEngine reject impossible placements; we only ensure the
      // metadata is consistent when a placement is otherwise plausible.
      return normalized;
    }

    const board = gameState.board;
    const posKey = positionToString(normalized.to);
    const stack = board.stacks.get(posKey as any);
    const isOccupied = !!stack && stack.rings.length > 0;

    if (isOccupied) {
      // Canonical rule: at most one ring per placement onto an existing
      // stack, and the placement is flagged as stacking.
      normalized.placedOnStack = true;
      normalized.placementCount = 1;
      return normalized;
    }

    // Empty cell: allow small multi-ring placements. If the service
    // already provided a placementCount, clamp it; otherwise choose a
    // simple count in [1, min(3, ringsInHand)].
    const maxPerPlacement = ringsInHand;
    if (maxPerPlacement <= 0) {
      return normalized;
    }

    if (normalized.placementCount && normalized.placementCount > 0) {
      const clamped = Math.min(Math.max(normalized.placementCount, 1), maxPerPlacement);
      normalized.placementCount = clamped;
      normalized.placedOnStack = false;
      return normalized;
    }

    const upper = Math.min(3, maxPerPlacement);
    const chosen = upper > 1 ? 1 + Math.floor(Math.random() * upper) : 1;
    normalized.placementCount = chosen;
    normalized.placedOnStack = false;

    return normalized;
  }

  /**
   * Evaluate a position from an AI's perspective via Python microservice
   */
  async evaluatePosition(playerNumber: number, gameState: GameState): Promise<number> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const response = await getAIServiceClient().evaluatePosition(gameState, playerNumber);

      logger.debug('Position evaluated', {
        playerNumber,
        score: response.score,
      });

      return response.score;
    } catch (error) {
      logger.error('Failed to evaluate position from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a line_reward_option for an AI-controlled
   * player. This is the service-backed analogue of the local heuristic in
   * AIInteractionHandler and keeps all remote AI behaviour behind this
   * façade.
   */
  async getLineRewardChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: LineRewardChoice['options']
  ): Promise<LineRewardChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const response = await getAIServiceClient().getLineRewardChoice(
        gameState,
        playerNumber,
        config.difficulty,
        aiType as unknown as ServiceAIType,
        options
      );

      logger.info('AI line_reward_option choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get line_reward_option choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a ring_elimination option for an
   * AI-controlled player. This mirrors the TypeScript
   * AIInteractionHandler heuristic (smallest capHeight, then
   * smallest totalHeight) but keeps the remote call behind this
   * façade so callers do not need to know about the Python
   * service directly.
   */
  async getRingEliminationChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: RingEliminationChoice['options']
  ): Promise<RingEliminationChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const response = await getAIServiceClient().getRingEliminationChoice(
        gameState,
        playerNumber,
        config.difficulty,
        aiType as unknown as ServiceAIType,
        options
      );

      logger.info('AI ring_elimination choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get ring_elimination choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a region_order option for an
   * AI-controlled player. This mirrors the TypeScript
   * AIInteractionHandler heuristic (largest region by size, with
   * additional context from GameState) but keeps the remote call
   * behind this façade so callers do not need to know about the
   * Python service directly.
   */
  async getRegionOrderChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: RegionOrderChoice['options']
  ): Promise<RegionOrderChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const response = await getAIServiceClient().getRegionOrderChoice(
        gameState,
        playerNumber,
        config.difficulty,
        aiType as unknown as ServiceAIType,
        options
      );

      logger.info('AI region_order choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get region_order choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Check if a player is controlled by AI
   */
  isAIPlayer(playerNumber: number): boolean {
    return this.aiConfigs.has(playerNumber);
  }

  /**
   * Get all AI player numbers
   */
  getAllAIPlayerNumbers(): number[] {
    return Array.from(this.aiConfigs.keys());
  }

  /**
   * Clear all AI players
   */
  clearAll(): void {
    this.aiConfigs.clear();
  }

  /**
   * Check AI service health
   */
  async checkServiceHealth(): Promise<boolean> {
    try {
      return await getAIServiceClient().healthCheck();
    } catch (error) {
      logger.error('AI service health check failed', { error });
      return false;
    }
  }

  /**
   * Clear AI service cache
   */
  async clearServiceCache(): Promise<void> {
    try {
      await getAIServiceClient().clearCache();
      logger.info('AI service cache cleared');
    } catch (error) {
      logger.error('Failed to clear AI service cache', { error });
      throw error;
    }
  }

  /**
   * Auto-select AI type based on difficulty level
   * @param difficulty - Difficulty level (1-10)
   * @returns Recommended AI type for this difficulty
   */
  private selectAITypeForDifficulty(difficulty: number): AIType {
    if (difficulty >= 1 && difficulty <= 2) {
      return AIType.RANDOM; // Levels 1-2: Random AI
    } else if (difficulty >= 3 && difficulty <= 5) {
      return AIType.HEURISTIC; // Levels 3-5: Heuristic AI
    } else if (difficulty >= 6 && difficulty <= 8) {
      return AIType.MINIMAX; // Levels 6-8: Minimax AI (falls back to Heuristic if not implemented)
    } else {
      return AIType.MCTS; // Levels 9-10: MCTS AI (falls back to Heuristic if not implemented)
    }
  }

  /** Map shared AITacticType values onto the internal AIType enum. */
  private mapAITacticToAIType(tactic: AITacticType): AIType {
    switch (tactic) {
      case 'random':
        return AIType.RANDOM;
      case 'heuristic':
        return AIType.HEURISTIC;
      case 'minimax':
        return AIType.MINIMAX;
      case 'mcts':
        return AIType.MCTS;
      default:
        return AIType.HEURISTIC;
    }
  }

  /** Map internal AIType to the shared AITacticType union. */
  private mapAITypeToTactic(type: AIType): AITacticType {
    switch (type) {
      case AIType.RANDOM:
        return 'random';
      case AIType.HEURISTIC:
        return 'heuristic';
      case AIType.MINIMAX:
        return 'minimax';
      case AIType.MCTS:
        return 'mcts';
      default:
        return 'heuristic';
    }
  }

  /**
   * Get AI description for difficulty level
   */
  static getAIDescription(difficulty: number): string {
    const descriptions: Record<number, string> = {
      1: 'Very Easy - Random moves with high error rate',
      2: 'Easy - Mostly random moves with some filtering',
      3: 'Medium-Easy - Basic strategy with occasional mistakes',
      4: 'Medium - Balanced play with tactical awareness',
      5: 'Medium-Hard - Strong tactical play',
      6: 'Hard - Advanced tactics and some planning',
      7: 'Very Hard - Deep planning and strong positional play',
      8: 'Expert - Excellent tactics and strategy',
      9: 'Master - Near-perfect play with deep calculation',
      10: 'Grandmaster - Optimal play across all phases',
    };

    return descriptions[difficulty] || 'Unknown difficulty level';
  }

  /**
   * Get recommended difficulty for player skill level
   */
  static getRecommendedDifficulty(
    skillLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert'
  ): number {
    const recommendations = {
      beginner: 2, // Easy
      intermediate: 4, // Medium
      advanced: 6, // Hard
      expert: 8, // Expert
    };

    return recommendations[skillLevel];
  }
}

/**
 * Singleton instance for global AI engine
 */
export const globalAIEngine = new AIEngine();
