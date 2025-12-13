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
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice,
  Position,
  positionToString,
} from '../../../shared/types/game';
import type { CancellationToken } from '../../../shared/utils/cancellation';
import { getAIServiceClient, AIType as ServiceAIType } from '../../services/AIServiceClient';
import { logger } from '../../utils/logger';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';
import {
  chooseLocalMoveFromCandidates as chooseSharedLocalMoveFromCandidates,
  LocalAIRng,
} from '../../../shared/engine/localAIMoveSelection';
import { SeededRNG } from '../../../shared/utils/rng';
import { aiMoveLatencyHistogram, aiFallbackCounter } from '../../utils/rulesParityMetrics';
import { getMetricsService } from '../../services/MetricsService';

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts',
  DESCENT = 'descent',
}

export interface AIConfig {
  difficulty: number;
  /** Optional search-budget hint in milliseconds. Must not be used to add artificial delay after a move is chosen. */
  thinkTime?: number;
  randomness?: number;
  /** Tactical engine chosen for this AI config. */
  aiType?: AIType;
  /** How this AI makes decisions about moves/choices. */
  mode?: AIControlMode;
}

/**
 * Lightweight per-player diagnostics for AI service usage. These counters are
 * incremented when the Python AI service fails or when the engine falls back
 * to local heuristics, and can be queried by tests and orchestration layers
 * to detect degraded AI quality modes.
 */
export interface AIDiagnostics {
  /** Number of times the AI service call failed for this player. */
  serviceFailureCount: number;
  /**
   * Number of times a local fallback move was generated because the AI
   * service was unavailable or failed. Note that this is counted at the
   * selection layer; downstream rules validation may still reject the move.
   */
  localFallbackCount: number;
}

/**
 * Canonical difficulty presets for the TypeScript backend. This table is kept
 * in lockstep with the Python service's difficulty ladder defined in
 * `ai-service/app/main.py` so that a given numeric difficulty corresponds to
 * the same underlying AI engine and coarse behaviour on both sides.
 *
 * Think times are conceptual per-move search budgets only. Hosts and engines
 * must never use `thinkTime` to insert artificial wall-clock delay after a
 * move has been selected; simpler engines are expected to ignore it entirely.
 */
export const AI_DIFFICULTY_PRESETS: Record<number, Partial<AIConfig> & { profileId: string }> = {
  1: {
    aiType: AIType.RANDOM,
    randomness: 0.5,
    thinkTime: 150,
    profileId: 'v1-random-1',
  },
  2: {
    aiType: AIType.HEURISTIC,
    randomness: 0.3,
    thinkTime: 200,
    profileId: 'v1-heuristic-2',
  },
  3: {
    aiType: AIType.MINIMAX,
    randomness: 0.15,
    thinkTime: 1800,
    profileId: 'v1-minimax-3',
  },
  4: {
    aiType: AIType.MINIMAX,
    randomness: 0.08,
    thinkTime: 2800,
    profileId: 'v1-minimax-4-nnue',
  },
  5: {
    aiType: AIType.MCTS,
    randomness: 0.05,
    thinkTime: 4000,
    profileId: 'v1-mcts-5',
  },
  6: {
    aiType: AIType.MCTS,
    randomness: 0.02,
    thinkTime: 5500,
    profileId: 'v1-mcts-6-neural',
  },
  7: {
    aiType: AIType.MCTS,
    randomness: 0.0,
    thinkTime: 7500,
    profileId: 'v1-mcts-7-neural',
  },
  8: {
    aiType: AIType.MCTS,
    randomness: 0.0,
    thinkTime: 9600,
    profileId: 'v1-mcts-8-neural',
  },
  9: {
    aiType: AIType.DESCENT,
    randomness: 0.0,
    thinkTime: 12600,
    profileId: 'v1-descent-9',
  },
  10: {
    aiType: AIType.DESCENT,
    randomness: 0.0,
    thinkTime: 16000,
    profileId: 'v1-descent-10',
  },
};

export class AIEngine {
  private aiConfigs: Map<number, AIConfig> = new Map();

  /**
   * Internal per-player diagnostics map keyed by playerNumber. This is kept
   * private to avoid accidental mutation; callers access a cloned snapshot
   * via getDiagnostics(...).
   */
  private diagnostics: Map<number, AIDiagnostics> = new Map();

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

    const basePreset = AI_DIFFICULTY_PRESETS[difficulty] ?? AI_DIFFICULTY_PRESETS[5];

    const aiType = profile.aiType
      ? this.mapAITacticToAIType(profile.aiType)
      : (basePreset.aiType ?? this.selectAITypeForDifficulty(difficulty));

    const config: AIConfig = {
      difficulty,
      aiType,
      mode: profile.mode ?? 'service',
    };

    if (typeof basePreset.randomness === 'number') {
      config.randomness = basePreset.randomness;
    }

    if (typeof basePreset.thinkTime === 'number') {
      config.thinkTime = basePreset.thinkTime;
    }

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
   * Get move from AI player via Python microservice.
   *
   * @param playerNumber - The player number
   * @param gameState - Current game state
   * @param rng - Optional RNG hook used by local fallback paths. When
   *   provided, this is threaded through to getLocalAIMove so that test
   *   harnesses and parity tools can keep sandbox and backend AI on the
   *   same deterministic RNG stream.
   */
  async getAIMove(
    playerNumber: number,
    gameState: GameState,
    rng?: LocalAIRng,
    options?: { token?: CancellationToken }
  ): Promise<Move | null> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    const metrics = getMetricsService();
    const requestStart = performance.now();
    const difficultyLabel = String(config.difficulty ?? 'n/a');
    const effectiveRng: LocalAIRng =
      rng ?? this.createDeterministicLocalRng(gameState, playerNumber);

    // Get valid moves for validation
    const boardManager = new BoardManager(gameState.boardType);
    const ruleEngine = new RuleEngine(boardManager, gameState.boardType);
    let validMoves = ruleEngine.getValidMoves(gameState);

    // Layer in the swap_sides meta-move (pie rule) for Player 2 when
    // enabled, mirroring backend GameEngine.shouldOfferSwapSidesMetaMove.
    if (this.shouldOfferSwapSidesForGameState(gameState)) {
      const alreadyHasSwap = validMoves.some((m) => m.type === 'swap_sides');

      if (!alreadyHasSwap) {
        const moveNumber = gameState.moveHistory.length + 1;
        const swapMove: Move = {
          id: `swap_sides-${moveNumber}`,
          type: 'swap_sides',
          player: 2,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        } as Move;

        validMoves = [...validMoves, swapMove];
      }
    }

    if (validMoves.length === 0) {
      logger.warn('No valid moves available for AI player', { playerNumber });
      return null;
    }

    // If only one move is available, return it immediately
    if (validMoves.length === 1) {
      logger.info('Single valid move for AI player', {
        playerNumber,
        moveType: validMoves[0].type,
      });
      return validMoves[0];
    }

    let lastError: Error | null = null;

    // Level 1: Try Python AI service (if mode is 'service')
    if (config.mode === 'service') {
      try {
        const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
        const serviceAIType = this.mapInternalTypeToServiceType(aiType);
        const response = await getAIServiceClient().getAIMove(
          gameState,
          playerNumber,
          config.difficulty,
          serviceAIType,
          undefined,
          options?.token ? { token: options.token } : undefined
        );

        const normalizedMove = this.normalizeServiceMove(response.move, gameState, playerNumber);

        // Validate that the AI service returned a move that's in the valid moves list
        if (normalizedMove) {
          const isValid = this.validateMoveInList(normalizedMove, validMoves);

          if (isValid) {
            logger.info('AI move generated via remote service', {
              playerNumber,
              moveType: normalizedMove.type,
              evaluation: response.evaluation,
              thinkingTime: response.thinking_time_ms,
              aiType: response.ai_type,
            });
            return normalizedMove;
          } else {
            logger.warn('Remote AI service returned invalid move', {
              playerNumber,
              suggestedMove: normalizedMove,
              validMoveCount: validMoves.length,
            });
            aiFallbackCounter.labels('validation_failed').inc();
            getMetricsService().recordAIFallback('validation_failed');
          }
        } else {
          // Service did not return a usable move; fall back to local heuristics.
          aiFallbackCounter.labels('no_move_from_service').inc();
          getMetricsService().recordAIFallback('no_move_from_service');
        }
      } catch (error) {
        lastError = error as Error;

        // Classify the failure reason using the structured aiErrorType set by
        // AIServiceClient so Prometheus can track why we fall back.
        const errorWithType = error as Error & { aiErrorType?: string };
        const errorType = errorWithType.aiErrorType;
        let reason: string;
        switch (errorType) {
          case 'connection_refused':
            reason = 'connection_refused';
            break;
          case 'timeout':
            reason = 'timeout';
            break;
          case 'service_unavailable':
            reason = 'service_unavailable';
            break;
          case 'server_error':
            reason = 'server_error';
            break;
          case 'client_error':
            reason = 'client_error';
            break;
          case 'overloaded':
            reason = 'overloaded';
            break;
          default:
            if (error instanceof Error && error.message.includes('Circuit breaker is open')) {
              reason = 'circuit_open';
            } else {
              reason = 'python_error';
            }
        }

        aiFallbackCounter.labels(reason).inc();
        getMetricsService().recordAIFallback(reason);

        logger.warn('Remote AI service failed, falling back to local heuristics', {
          error: error instanceof Error ? error.message : 'Unknown error',
          playerNumber,
          difficulty: config.difficulty,
        });

        // Record the service failure for diagnostics
        const diag = this.getOrCreateDiagnostics(playerNumber);
        diag.serviceFailureCount += 1;
      }
    }

    // Level 2: Local heuristic AI
    try {
      const heuristicStart = performance.now();
      const localMove = this.selectLocalHeuristicMove(gameState, validMoves, effectiveRng);

      if (localMove) {
        const duration = performance.now() - heuristicStart;
        aiMoveLatencyHistogram.labels('heuristic', difficultyLabel).observe(duration);
        const totalDuration = performance.now() - requestStart;
        metrics.recordAIRequest('fallback');
        metrics.recordAIRequestLatencyMs(totalDuration, 'fallback');

        const diag = this.getOrCreateDiagnostics(playerNumber);
        diag.localFallbackCount += 1;

        logger.info('AI move selected via local heuristics (fallback)', {
          playerNumber,
          moveType: localMove.type,
          fallback: config.mode === 'service',
        });
        return localMove;
      }
    } catch (error) {
      lastError = error as Error;
      logger.error('Local heuristic AI failed, falling back to random', {
        error: error instanceof Error ? error.message : 'Unknown error',
        playerNumber,
      });
    }

    // Level 3: Random selection (last resort)
    logger.warn('All AI methods failed, selecting random valid move', {
      playerNumber,
      validMoveCount: validMoves.length,
      lastError: lastError?.message,
    });

    const totalDuration = performance.now() - requestStart;
    metrics.recordAIRequest('fallback');
    metrics.recordAIRequestLatencyMs(totalDuration, 'fallback');

    const randomIndex = Math.floor(effectiveRng() * validMoves.length);
    return validMoves[randomIndex];
  }

  /**
   * Determine whether the given GameState should expose a swap_sides
   * meta-move (pie rule) for Player 2, mirroring the gate in
   * GameEngine.shouldOfferSwapSidesMetaMove.
   */
  private shouldOfferSwapSidesForGameState(state: GameState): boolean {
    // Config gating: swap_sides is only offered when explicitly enabled
    // via state.rulesOptions.swapRuleEnabled. When the flag is absent or
    // false, the pie rule is considered disabled.
    if (!state.rulesOptions?.swapRuleEnabled) {
      return false;
    }

    if (state.gameStatus !== 'active') return false;
    if (state.players.length !== 2) return false;
    if (state.currentPlayer !== 2) return false;

    // Only in interactive phases.
    if (
      state.currentPhase !== 'ring_placement' &&
      state.currentPhase !== 'movement' &&
      state.currentPhase !== 'capture' &&
      state.currentPhase !== 'chain_capture'
    ) {
      return false;
    }

    if (state.moveHistory.length === 0) return false;

    const hasSwapMove = state.moveHistory.some((m) => m.type === 'swap_sides');
    if (hasSwapMove) return false;

    const hasP1Move = state.moveHistory.some((m) => m.player === 1);
    const hasP2Move = state.moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides');

    // Exactly: at least one move from P1, none from P2 yet (excluding swap_sides).
    return hasP1Move && !hasP2Move;
  }

  /**
   * Validate that a move is in the list of valid moves
   */
  private validateMoveInList(move: Move, validMoves: Move[]): boolean {
    return validMoves.some((validMove) => this.movesEqual(validMove, move));
  }

  /**
   * Deep equality check for Move objects
   */
  private movesEqual(m1: Move, m2: Move): boolean {
    // Compare essential move properties
    if (m1.type !== m2.type) return false;
    if (m1.player !== m2.player) return false;

    // Compare positions if they exist
    if (m1.from || m2.from) {
      if (!m1.from || !m2.from) return false;
      if (!this.positionsEqual(m1.from, m2.from)) return false;
    }

    if (m1.to || m2.to) {
      if (!m1.to || !m2.to) return false;
      if (!this.positionsEqual(m1.to, m2.to)) return false;
    }

    // Compare capture target if exists
    if (m1.captureTarget || m2.captureTarget) {
      if (!m1.captureTarget || !m2.captureTarget) return false;
      if (!this.positionsEqual(m1.captureTarget, m2.captureTarget)) return false;
    }

    // Compare placement count for place_ring moves
    if (m1.type === 'place_ring' && m2.type === 'place_ring') {
      if (m1.placementCount !== m2.placementCount) return false;
    }

    return true;
  }

  /**
   * Compare two positions for equality
   */
  private positionsEqual(p1: Position, p2: Position): boolean {
    if (p1.x !== p2.x || p1.y !== p2.y) return false;
    // Handle hexagonal z coordinate
    if (p1.z !== undefined || p2.z !== undefined) {
      return p1.z === p2.z;
    }
    return true;
  }

  /**
   * Select a move using local heuristics from the valid moves list
   */
  private selectLocalHeuristicMove(
    gameState: GameState,
    validMoves: Move[],
    rng: LocalAIRng
  ): Move | null {
    if (validMoves.length === 0) {
      return null;
    }

    // Use the shared chooseLocalMoveFromCandidates for consistent heuristics
    return chooseSharedLocalMoveFromCandidates(gameState.currentPlayer, gameState, validMoves, rng);
  }

  /**
   * Create a deterministic RNG for local fallback paths when callers do not
   * supply an explicit RNG. This RNG is derived solely from the per-game
   * GameState.rngSeed and the player number so that for a fixed seed and game
   * configuration, local fallback behaviour is reproducible.
   */
  private createDeterministicLocalRng(gameState: GameState, playerNumber: number): LocalAIRng {
    const baseSeed = typeof gameState.rngSeed === 'number' ? gameState.rngSeed : 0;
    const mixed = (baseSeed ^ (playerNumber * 0x9e3779b1)) >>> 0;
    const seeded = new SeededRNG(mixed);
    return () => seeded.next();
  }

  /**
   * Generate a move using local heuristics when the AI service is unavailable.
   * Uses RuleEngine to find valid moves and selects one randomly. The
   * optional rng parameter allows test harnesses and parity tools to share
   * a deterministic RNG stream with other AI callers (e.g. sandbox AI).
   */
  private getLocalAIMove(playerNumber: number, gameState: GameState, rng: LocalAIRng): Move | null {
    try {
      const boardManager = new BoardManager(gameState.boardType);
      const ruleEngine = new RuleEngine(boardManager, gameState.boardType);

      let validMoves = ruleEngine.getValidMoves(gameState);

      // Enforce the canonical "must move placed stack" rule when available.
      // GameEngine/TurnEngine track the origin stack via mustMoveFromStackKey;
      // RuleEngine itself is stateless, so we need to apply this constraint
      // here before handing moves to the shared local-selection policy.
      if (
        gameState.currentPhase === 'movement' ||
        gameState.currentPhase === 'capture' ||
        gameState.currentPhase === 'chain_capture'
      ) {
        const mustMoveFromStackKey = gameState.mustMoveFromStackKey;

        if (mustMoveFromStackKey) {
          validMoves = validMoves.filter((m) => {
            const isMovementOrCaptureMove =
              m.type === 'move_stack' ||
              m.type === 'move_ring' ||
              m.type === 'build_stack' ||
              m.type === 'overtaking_capture' ||
              m.type === 'continue_capture_segment';

            if (!isMovementOrCaptureMove) {
              return true;
            }

            if (!m.from) {
              return false;
            }

            return positionToString(m.from) === mustMoveFromStackKey;
          });
        } else {
          // Backwards-compatible fallback for older fixtures/states that do
          // not populate mustMoveFromStackKey: infer the must-move origin
          // from the last place_ring move by the current player.
          const lastMove = gameState.moveHistory[gameState.moveHistory.length - 1];

          if (
            lastMove &&
            lastMove.type === 'place_ring' &&
            lastMove.player === gameState.currentPlayer &&
            lastMove.to
          ) {
            const placedKey = positionToString(lastMove.to);
            validMoves = validMoves.filter((m) => {
              const isMovementOrCaptureMove =
                m.type === 'move_stack' ||
                m.type === 'move_ring' ||
                m.type === 'overtaking_capture' ||
                m.type === 'build_stack' ||
                m.type === 'continue_capture_segment';

              if (!isMovementOrCaptureMove) {
                return true;
              }

              return m.from && positionToString(m.from) === placedKey;
            });
          }
        }
      }

      if (validMoves.length === 0) {
        return null;
      }

      // Delegate to the shared selection policy so that local fallback and
      // any external harnesses can share the same move preferences. When
      // an explicit rng is provided, it is threaded through so parity
      // harnesses can keep backend and sandbox AI on the same RNG stream.
      return this.chooseLocalMoveFromCandidates(playerNumber, gameState, validMoves, rng);
    } catch (error) {
      logger.error('Failed to generate local AI move', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      return null;
    }
  }

  /**
   * Public wrapper around the local heuristic move generator used by the
   * service fallback path. This allows orchestrators such as GameSession
   * to explicitly request a purely local move when the Python AI service
   * has failed or produced invalid moves, without re-entering the service
   * code path.
   */
  public getLocalFallbackMove(
    playerNumber: number,
    gameState: GameState,
    rng?: LocalAIRng
  ): Move | null {
    const effectiveRng = rng ?? this.createDeterministicLocalRng(gameState, playerNumber);
    const move = this.getLocalAIMove(playerNumber, gameState, effectiveRng);

    if (move) {
      const diag = this.getOrCreateDiagnostics(playerNumber);
      diag.localFallbackCount += 1;

      // Record that this move was generated via the explicit local fallback
      // path after service degradation (e.g. repeated Python AI failures or
      // rules-engine rejections of service moves).
      aiFallbackCounter.labels('service_degraded').inc();
    }

    return move;
  }

  /**
   * Shared local-selection policy used by the fallback path and test
   * harnesses. Given a set of already-legal moves (typically from
   * GameEngine.getValidMoves or RuleEngine.getValidMoves), prefer moves
   * that are more likely to make structural progress before falling back
   * to a pure random choice.
   */
  public chooseLocalMoveFromCandidates(
    playerNumber: number,
    gameState: GameState,
    candidates: Move[],
    rng: LocalAIRng
  ): Move | null {
    const selectedMove = chooseSharedLocalMoveFromCandidates(
      playerNumber,
      gameState,
      candidates,
      rng
    );

    if (selectedMove) {
      logger.info('Local AI fallback move generated', {
        playerNumber,
        moveType: selectedMove.type,
      });
    }

    return selectedMove;
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
    if ((normalized.type as string) === 'place') {
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
    const stack = board.stacks.get(posKey);
    const isOccupied = !!stack && stack.rings.length > 0;

    if (isOccupied) {
      // Canonical rule: at most one ring per placement onto an existing
      // stack, and the placement is flagged as stacking.
      normalized.placedOnStack = true;
      normalized.placementCount = 1;
      return normalized;
    }

    // Empty cell: allow small multi-ring placements. If the service
    // already provided a placementCount, clamp it; otherwise fall back
    // to a deterministic default of 1 ring. This avoids introducing any
    // additional RNG at the AI–rules boundary while keeping older
    // service versions (that omit placementCount) working.
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

    // If placementCount is missing for an otherwise valid empty-cell
    // placement, record this so we can add metrics and tighten the
    // contract in a future phase.
    logger.warn('AI service omitted placementCount for empty-cell place_ring; defaulting to 1', {
      playerNumber,
      position: normalized.to && positionToString(normalized.to),
      ringsInHand,
    });

    normalized.placementCount = 1;
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
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getLineRewardChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
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
    options: RingEliminationChoice['options'],
    requestOptions?: { token?: CancellationToken }
  ): Promise<RingEliminationChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getRingEliminationChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options,
        requestOptions
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
    options: RegionOrderChoice['options'],
    requestOptions?: { token?: CancellationToken }
  ): Promise<RegionOrderChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getRegionOrderChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options,
        requestOptions
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
   * Ask the AI service to choose a line_order option for an
   * AI-controlled player. This mirrors the TypeScript
   * AIInteractionHandler heuristic (prefer longest line) but keeps
   * the remote call behind this façade so callers do not need to
   * know about the Python service directly.
   */
  async getLineOrderChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: LineOrderChoice['options'],
    requestOptions?: { token?: CancellationToken }
  ): Promise<LineOrderChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getLineOrderChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options,
        requestOptions
      );

      logger.info('AI line_order choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get line_order choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a capture_direction option for an
   * AI-controlled player. This mirrors the TypeScript
   * AIInteractionHandler heuristic (prefer highest capturedCapHeight,
   * then central landingPosition) but keeps the remote call behind
   * this façade.
   */
  async getCaptureDirectionChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: CaptureDirectionChoice['options'],
    requestOptions?: { token?: CancellationToken }
  ): Promise<CaptureDirectionChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getCaptureDirectionChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options,
        requestOptions
      );

      logger.info('AI capture_direction choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get capture_direction choice from service', {
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
    this.diagnostics.clear();
  }

  /**
   * Get a snapshot of diagnostics for a given AI-controlled player. The
   * returned object is a shallow clone so callers cannot mutate the
   * internal counters directly.
   */
  getDiagnostics(playerNumber: number): AIDiagnostics | undefined {
    const diag = this.diagnostics.get(playerNumber);
    return diag
      ? {
          serviceFailureCount: diag.serviceFailureCount,
          localFallbackCount: diag.localFallbackCount,
        }
      : undefined;
  }

  /**
   * Internal helper: ensure a diagnostics record exists for the given
   * player and return it.
   */
  private getOrCreateDiagnostics(playerNumber: number): AIDiagnostics {
    let diag = this.diagnostics.get(playerNumber);
    if (!diag) {
      diag = { serviceFailureCount: 0, localFallbackCount: 0 };
      this.diagnostics.set(playerNumber, diag);
    }
    return diag;
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
   * Auto-select AI type based on difficulty level.
   *
   * This is a thin wrapper over the canonical AI_DIFFICULTY_PRESETS table so
   * that both Python and TypeScript resolve difficulty→engine in the same way.
   */
  private selectAITypeForDifficulty(difficulty: number): AIType {
    const clamped = difficulty < 1 ? 1 : difficulty > 10 ? 10 : difficulty;
    const preset = AI_DIFFICULTY_PRESETS[clamped];
    return preset.aiType ?? AIType.HEURISTIC;
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
      case 'descent':
        return AIType.DESCENT;
      default: {
        // Exhaustive check so that adding a new AITacticType forces this
        // mapping to be updated.
        const exhaustiveCheck: never = tactic;
        throw new Error(`Unhandled AITacticType in mapAITacticToAIType: ${exhaustiveCheck}`);
      }
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
      case AIType.DESCENT:
        return 'descent';
      default: {
        // Exhaustive check so that adding a new AIType forces this mapping
        // (and downstream service wiring) to be updated.
        const exhaustiveCheck: never = type;
        throw new Error(`Unhandled AIType in mapAITypeToTactic: ${exhaustiveCheck}`);
      }
    }
  }

  /**
   * Map the internal AIType enum used by the server onto the AIType enum
   * understood by the Python AI service. This indirection keeps the
   * wire-level contract stable even if the server or service introduce
   * additional implementation-specific variants in future.
   */
  private mapInternalTypeToServiceType(type: AIType): ServiceAIType {
    switch (type) {
      case AIType.RANDOM:
        return ServiceAIType.RANDOM;
      case AIType.HEURISTIC:
        return ServiceAIType.HEURISTIC;
      case AIType.MINIMAX:
        return ServiceAIType.MINIMAX;
      case AIType.MCTS:
        return ServiceAIType.MCTS;
      case AIType.DESCENT:
        return ServiceAIType.DESCENT;
      default: {
        const exhaustiveCheck: never = type;
        throw new Error(`Unhandled AIType in mapInternalTypeToServiceType: ${exhaustiveCheck}`);
      }
    }
  }

  /**
   * Get a human-friendly description for a given difficulty level.
   *
   * This is intentionally coarse-grained and should stay in sync with the
   * canonical ladder described in AI_ARCHITECTURE.md and
   * AI_DIFFICULTY_PRESETS above. The wording is kept honest about actual
   * strength (no promises of "perfect" or "optimal" play).
   */
  static getAIDescription(difficulty: number): string {
    const descriptions: Record<number, string> = {
      1: 'Level 1 – Beginner: Random AI that plays legal but weak moves',
      2: 'Level 2 – Easy: Heuristic AI with simple patterns and clear weaknesses',
      3: 'Level 3 – Minimax: Shallow search that sees basic tactics',
      4: 'Level 4 – Minimax: Deeper search with more consistent tactics',
      5: 'Level 5 – Minimax: Solid tactical play with occasional oversights',
      6: 'Level 6 – Minimax: Strong tactical play and some planning ahead',
      7: 'Level 7 – MCTS: Expert search that samples many futures positions',
      8: 'Level 8 – MCTS: Strong expert play with robust search in complex boards',
      9: 'Level 9 – Descent: Hybrid MCTS/NN engine aimed at very strong play',
      10: 'Level 10 – Descent: Strongest available engine; very challenging but not perfect',
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
