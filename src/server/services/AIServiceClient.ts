/**
 * Client for communicating with the Python AI Service
 * Makes HTTP requests to the FastAPI microservice for AI move and choice selection.
 * Includes circuit breaker pattern for resilience.
 * Integrated with ServiceStatusManager for graceful degradation.
 */

import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import type { CancellationToken } from '../../shared/utils/cancellation';
import { getMetricsService } from './MetricsService';
import {
  GameState,
  Move,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  LineOrderChoice,
  CaptureDirectionChoice,
  BoardType,
} from '../../shared/types/game';
import { logger } from '../utils/logger';
import { aiMoveLatencyHistogram } from '../utils/rulesParityMetrics';
import { getServiceStatusManager } from './ServiceStatusManager';
import type { PositionEvaluationByPlayer } from '../../shared/types/websocket';

/**
 * High-level error codes surfaced by the AI service client. These are used
 * by upstream layers (HTTP handlers, WebSocket orchestration, tests) to map
 * dependency failures into predictable 5xx responses and fallback paths.
 */
export type AIServiceErrorCode =
  | 'AI_SERVICE_TIMEOUT'
  | 'AI_SERVICE_UNAVAILABLE'
  | 'AI_SERVICE_ERROR'
  | 'AI_SERVICE_OVERLOADED';

/**
 * Circuit breaker to prevent hammering a failing AI service
 */
class CircuitBreaker {
  private failureCount = 0;
  private lastFailureTime = 0;
  private isOpen = false;
  private readonly threshold = 5; // failures before opening
  private readonly timeout = 60000; // 1 minute cooldown

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.isOpen) {
      const now = Date.now();
      if (now - this.lastFailureTime > this.timeout) {
        this.reset(); // Try again after timeout
        logger.info('Circuit breaker transitioning from open to half-open');
      } else {
        throw new Error('Circuit breaker is open - AI service temporarily unavailable');
      }
    }

    try {
      const result = await fn();
      this.reset();
      return result;
    } catch (error) {
      this.recordFailure();
      throw error;
    }
  }

  private recordFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    if (this.failureCount >= this.threshold) {
      this.isOpen = true;
      logger.warn('Circuit breaker opened after repeated failures', {
        failureCount: this.failureCount,
        threshold: this.threshold,
      });
    }
  }

  private reset(): void {
    if (this.failureCount > 0 || this.isOpen) {
      logger.info('Circuit breaker reset', {
        previousFailures: this.failureCount,
        wasOpen: this.isOpen,
      });
    }
    this.failureCount = 0;
    this.isOpen = false;
  }

  getStatus(): { isOpen: boolean; failureCount: number } {
    return {
      isOpen: this.isOpen,
      failureCount: this.failureCount,
    };
  }

  /**
   * Check if the circuit breaker is currently open.
   */
  isCircuitOpen(): boolean {
    if (this.isOpen) {
      const now = Date.now();
      if (now - this.lastFailureTime > this.timeout) {
        return false; // Would transition to half-open
      }
      return true;
    }
    return false;
  }
}

export interface AIConfig {
  difficulty: number;
  thinkTime?: number;
  randomness?: number;
}

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts',
  DESCENT = 'descent',
}

export interface MoveRequest {
  game_state: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  seed?: number;
}

export interface MoveResponse {
  move: Move | null;
  evaluation: number;
  thinking_time_ms: number;
  ai_type: string;
  difficulty: number;
}

export interface EvaluationRequest {
  game_state: GameState;
  player_number: number;
}

export interface EvaluationResponse {
  score: number;
  breakdown: Record<string, number>;
}

export interface PositionEvaluationApiResponse {
  engine_profile: string;
  board_type: BoardType | string;
  game_id: string;
  move_number: number;
  per_player: Record<number, PositionEvaluationByPlayer>;
  evaluation_scale: 'zero_sum_margin' | 'win_probability';
  generated_at: string;
}

export interface LineRewardChoiceRequestPayload {
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: LineRewardChoice['options'];
}

export interface LineRewardChoiceResponsePayload {
  selectedOption: LineRewardChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface RingEliminationChoiceRequestPayload {
  // Optional for now so callers can omit GameState while we
  // progressively adopt full-game-state-aware heuristics on the
  // Python side.
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: RingEliminationChoice['options'];
}

export interface RingEliminationChoiceResponsePayload {
  selectedOption: RingEliminationChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface RegionOrderChoiceRequestPayload {
  // Optional for now so callers can omit GameState while we
  // progressively adopt full-game-state-aware heuristics on the
  // Python side.
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: RegionOrderChoice['options'];
}

export interface RegionOrderChoiceResponsePayload {
  selectedOption: RegionOrderChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface LineOrderChoiceRequestPayload {
  // Optional GameState for future context-aware heuristics.
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: LineOrderChoice['options'];
}

export interface LineOrderChoiceResponsePayload {
  selectedOption: LineOrderChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface CaptureDirectionChoiceRequestPayload {
  // Optional GameState for future context-aware heuristics.
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: CaptureDirectionChoice['options'];
}

export interface CaptureDirectionChoiceResponsePayload {
  selectedOption: CaptureDirectionChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface AIServiceRequestOptions {
  /** Optional cooperative cancellation token for this request. */
  token?: CancellationToken;
}

/**
 * Client for interacting with the Python AI microservice.
 * Includes circuit breaker for resilience and timeout handling.
 */
export class AIServiceClient {
  private client: AxiosInstance;
  private baseURL: string;
  private circuitBreaker: CircuitBreaker;

  // Node-local concurrency counters used to provide basic backpressure for
  // AI-heavy workloads. These are intentionally static so that all instances
  // of AIServiceClient within a single Node process share the same cap.
  private static inFlightRequests = 0;
  // Max concurrent AI HTTP calls per Node instance.
  // Config-driven so operators can tune without code changes.
  private static maxConcurrent: number =
    (config as { aiService?: { maxConcurrent?: number } }).aiService?.maxConcurrent ?? 16;

  private static incrementConcurrency(): boolean {
    if (AIServiceClient.inFlightRequests >= AIServiceClient.maxConcurrent) {
      return false;
    }
    AIServiceClient.inFlightRequests += 1;
    return true;
  }

  private static decrementConcurrency(): void {
    if (AIServiceClient.inFlightRequests > 0) {
      AIServiceClient.inFlightRequests -= 1;
    }
  }

  /**
   * Exposed for tests to observe the current concurrency level without
   * coupling production code to Prometheus or other metrics sinks.
   */
  static getInFlightRequestsForTest(): number {
    return AIServiceClient.inFlightRequests;
  }

  constructor(baseURL?: string) {
    this.baseURL = baseURL || config.aiService.url;
    this.circuitBreaker = new CircuitBreaker();

    this.client = axios.create({
      baseURL: this.baseURL,
      // Enforce a bounded per-request timeout for all AI interactions so that
      // slow or unavailable dependencies surface promptly and upstream layers
      // can apply fallbacks instead of hanging indefinitely.
      timeout: config.aiService.requestTimeoutMs,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // Enhanced error logging with categorization
        const errorType = this.categorizeError(error);
        // Attach categorized error type so downstream callers (e.g. AIEngine)
        // can emit structured fallback metrics without depending on axios internals.
        (error as Error & { aiErrorType?: string }).aiErrorType = errorType;

        logger.error('AI Service error:', {
          type: errorType,
          message: error.message,
          response: error.response?.data,
          status: error.response?.status,
          code: error.code,
        });
        throw error;
      }
    );
  }

  /**
   * Categorize error type for better diagnostics
   */
  private categorizeError(error: unknown): string {
    const err = error as { code?: string; response?: { status?: number } };
    if (err.code === 'ECONNREFUSED') return 'connection_refused';
    if (err.code === 'ETIMEDOUT' || err.code === 'ECONNABORTED') return 'timeout';
    if (err.response?.status === 500) return 'server_error';
    if (err.response?.status === 503) return 'service_unavailable';
    if (err.response?.status && err.response.status >= 400 && err.response.status < 500)
      return 'client_error';
    return 'unknown';
  }

  /**
   * Get AI-selected move for current game state with circuit breaker protection.
   */
  async getAIMove(
    gameState: GameState,
    playerNumber: number,
    difficulty: number = 5,
    aiType?: AIType,
    seed?: number,
    options?: AIServiceRequestOptions
  ): Promise<MoveResponse> {
    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');
    const metrics = getMetricsService();

    // Cooperative pre-flight cancellation: if a token is provided and is
    // already canceled, avoid touching the circuit breaker or HTTP layer.
    options?.token?.throwIfCanceled('before dispatching AIServiceClient.getAIMove');

    // Fast-fail when the node-local concurrency cap has been reached so that
    // AI-heavy workloads cannot starve the service or other games.
    if (!AIServiceClient.incrementConcurrency()) {
      const overloadedError: Error & {
        statusCode?: number;
        code?: AIServiceErrorCode | 'AI_SERVICE_OVERLOADED';
        isOperational?: boolean;
        aiErrorType?: string;
      } = new Error('AI service overloaded: too many concurrent requests');

      overloadedError.statusCode = 503;
      overloadedError.code = 'AI_SERVICE_OVERLOADED';
      overloadedError.isOperational = true;
      // Distinct aiErrorType so AIEngine can emit ai_fallback_total{reason="overloaded"}.
      overloadedError.aiErrorType = 'overloaded';

      logger.warn('AI Service concurrency limit reached, rejecting request', {
        playerNumber,
        difficulty,
        maxConcurrent: AIServiceClient['maxConcurrent'],
      });

      const durationMs = performance.now() - startTime;
      metrics.recordAIRequest('error');
      metrics.recordAIRequestLatencyMs(durationMs, 'error');

      throw overloadedError;
    }

    try {
      return await this.circuitBreaker.execute(async () => {
        try {
          // Derive seed from gameState if not explicitly provided
          const effectiveSeed = seed ?? gameState.rngSeed;

          const request: MoveRequest = {
            game_state: gameState,
            player_number: playerNumber,
            difficulty,
            ...(aiType && { ai_type: aiType }),
            ...(effectiveSeed !== undefined && { seed: effectiveSeed }),
          };

          logger.info('Requesting AI move', {
            playerNumber,
            difficulty,
            aiType,
            phase: gameState.currentPhase,
          });

          const response = await this.client.post<MoveResponse>('/ai/move', request);
          const duration = performance.now() - startTime;
          const durationSeconds = duration / 1000;

          // Record latency for successful Python-service-backed move selection.
          aiMoveLatencyHistogram.labels('python', difficultyLabel).observe(duration);
          metrics.recordAIRequest('success');
          metrics.recordAIRequestDuration('python', difficultyLabel, durationSeconds);
          metrics.recordAIRequestLatencyMs(duration, 'success');

          // Update ServiceStatusManager on success
          this.updateServiceStatus('healthy', undefined, Math.round(duration));

          logger.info('AI move received', {
            aiType: response.data.ai_type,
            thinkingTime: response.data.thinking_time_ms,
            evaluation: response.data.evaluation,
            latencyMs: Math.round(duration),
          });

          return response.data;
        } catch (error) {
          const duration = performance.now() - startTime;
          const durationSeconds = duration / 1000;

          // Preserve the low-level error classification set by the axios
          // interceptor so AIEngine and observability layers can distinguish
          // between timeouts, connection failures, and other errors.
          const errorWithType = error as Error & { aiErrorType?: string };
          const aiErrorType = errorWithType.aiErrorType;

          logger.error('Failed to get AI move', {
            error,
            latencyMs: Math.round(duration),
            playerNumber,
            difficulty,
            aiErrorType,
          });

          const latencyOutcome: 'success' | 'fallback' | 'timeout' | 'error' =
            aiErrorType === 'timeout' ? 'timeout' : 'error';
          metrics.recordAIRequest('error');
          metrics.recordAIRequestDuration('python', difficultyLabel, durationSeconds);
          metrics.recordAIRequestLatencyMs(duration, latencyOutcome);
          if (aiErrorType === 'timeout') {
            metrics.recordAIRequestTimeout();
          }

          const structuredError: Error & {
            statusCode?: number;
            code?: AIServiceErrorCode;
            isOperational?: boolean;
            aiErrorType?: string;
          } = new Error(
            `AI Service failed to generate move: ${
              error instanceof Error ? error.message : 'Unknown error'
            }`
          );

          // Map transport-level/category details onto a stable, high-level code
          // so HTTP handlers and WebSocket flows can surface a predictable 5xx.
          if (aiErrorType === 'timeout') {
            structuredError.statusCode = 503;
            structuredError.code = 'AI_SERVICE_TIMEOUT';
          } else if (
            aiErrorType === 'connection_refused' ||
            aiErrorType === 'service_unavailable'
          ) {
            structuredError.statusCode = 503;
            structuredError.code = 'AI_SERVICE_UNAVAILABLE';
          } else {
            structuredError.statusCode = 502;
            structuredError.code = 'AI_SERVICE_ERROR';
          }

          // Mark as operational so the central error handler can treat this as a
          // handled dependency failure rather than an unexpected crash.
          structuredError.isOperational = true;

          // Preserve the fine-grained aiErrorType hint for AIEngine fallback
          // metrics (e.g. ai_fallback_total{reason="timeout"}).
          if (aiErrorType) {
            structuredError.aiErrorType = aiErrorType;
          }

          // Update ServiceStatusManager on failure
          this.updateServiceStatus(
            aiErrorType === 'timeout' ? 'degraded' : 'unhealthy',
            structuredError.message,
            Math.round(performance.now() - startTime)
          );

          throw structuredError;
        }
      });
    } finally {
      AIServiceClient.decrementConcurrency();
    }
  }

  /**
   * Get circuit breaker status for monitoring
   */
  getCircuitBreakerStatus(): { isOpen: boolean; failureCount: number } {
    return this.circuitBreaker.getStatus();
  }

  /**
   * Evaluate current position from a player's perspective.
   */
  async evaluatePosition(
    gameState: GameState,
    playerNumber: number,
    options?: AIServiceRequestOptions
  ): Promise<EvaluationResponse> {
    const startTime = performance.now();

    // Cooperative pre-flight cancellation; position evaluation is a pure
    // dependency call and should not mutate state when canceled.
    options?.token?.throwIfCanceled('before dispatching AIServiceClient.evaluatePosition');

    try {
      const request: EvaluationRequest = {
        game_state: gameState,
        player_number: playerNumber,
      };

      const response = await this.client.post<EvaluationResponse>('/ai/evaluate', request);
      const duration = performance.now() - startTime;

      logger.debug('Position evaluated', {
        playerNumber,
        score: response.data.score,
        breakdown: response.data.breakdown,
        latencyMs: Math.round(duration),
      });

      return response.data;
    } catch (error) {
      const duration = performance.now() - startTime;
      logger.error('Failed to evaluate position', {
        error,
        latencyMs: Math.round(duration),
      });
      throw new Error(
        `AI Service failed to evaluate position: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Evaluate the current position from all players' perspectives using the
   * analysis-mode /ai/evaluate_position endpoint.
   */
  async evaluatePositionMulti(
    gameState: GameState,
    options?: AIServiceRequestOptions
  ): Promise<PositionEvaluationApiResponse> {
    const startTime = performance.now();

    // Cooperative pre-flight cancellation; this is a pure analysis call.
    options?.token?.throwIfCanceled('before dispatching AIServiceClient.evaluatePositionMulti');

    try {
      const request = {
        game_state: gameState,
      };

      const response = await this.client.post<PositionEvaluationApiResponse>(
        '/ai/evaluate_position',
        request
      );
      const duration = performance.now() - startTime;

      logger.debug('Multi-player position evaluated', {
        gameId: gameState.id,
        boardType: gameState.boardType,
        moveNumber: response.data.move_number,
        latencyMs: Math.round(duration),
      });

      return response.data;
    } catch (error) {
      const duration = performance.now() - startTime;
      logger.error('Failed to evaluate position (multi)', {
        error,
        latencyMs: Math.round(duration),
      });
      throw new Error(
        `AI Service failed to evaluate position (multi): ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected line reward option for a LineRewardChoice.
   *
   * For now this mirrors the TypeScript AIInteractionHandler heuristic by
   * preferring Option 2 when available, but delegates the decision to the
   * Python service to keep all AI behaviour behind a single façade.
   */
  async getLineRewardChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: LineRewardChoice['options'],
    requestOptions?: AIServiceRequestOptions
  ): Promise<LineRewardChoiceResponsePayload> {
    // Cooperative pre-flight cancellation; choice selection is a pure
    // dependency call and should not proceed when canceled.
    requestOptions?.token?.throwIfCanceled(
      'before dispatching AIServiceClient.getLineRewardChoice'
    );

    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');
    const metrics = getMetricsService();

    try {
      const request: LineRewardChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI line_reward_option choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<LineRewardChoiceResponsePayload>(
        '/ai/choice/line_reward_option',
        request
      );

      const duration = performance.now() - startTime;
      metrics.recordAIRequest('success');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, 'success');

      logger.info('AI line_reward_option choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get line_reward_option choice from AI service', {
        playerNumber,
        error,
      });
      const duration = performance.now() - startTime;
      const errorWithType = error as Error & { aiErrorType?: string };
      const aiErrorType = errorWithType.aiErrorType;
      const latencyOutcome: 'success' | 'fallback' | 'timeout' | 'error' =
        aiErrorType === 'timeout' ? 'timeout' : 'error';
      metrics.recordAIRequest('error');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, latencyOutcome);
      if (aiErrorType === 'timeout') {
        metrics.recordAIRequestTimeout();
      }

      throw new Error(
        `AI Service failed to choose line_reward_option: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected ring elimination option for a RingEliminationChoice.
   */
  async getRingEliminationChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: RingEliminationChoice['options'],
    requestOptions?: AIServiceRequestOptions
  ): Promise<RingEliminationChoiceResponsePayload> {
    // Cooperative pre-flight cancellation; choice selection is a pure
    // dependency call and should not proceed when canceled.
    requestOptions?.token?.throwIfCanceled(
      'before dispatching AIServiceClient.getRingEliminationChoice'
    );

    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');
    const metrics = getMetricsService();

    try {
      const request: RingEliminationChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI ring_elimination choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<RingEliminationChoiceResponsePayload>(
        '/ai/choice/ring_elimination',
        request
      );

      const duration = performance.now() - startTime;
      metrics.recordAIRequest('success');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, 'success');

      logger.info('AI ring_elimination choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get ring_elimination choice from AI service', {
        playerNumber,
        error,
      });
      const duration = performance.now() - startTime;
      const errorWithType = error as Error & { aiErrorType?: string };
      const aiErrorType = errorWithType.aiErrorType;
      const latencyOutcome: 'success' | 'fallback' | 'timeout' | 'error' =
        aiErrorType === 'timeout' ? 'timeout' : 'error';
      metrics.recordAIRequest('error');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, latencyOutcome);
      if (aiErrorType === 'timeout') {
        metrics.recordAIRequestTimeout();
      }

      throw new Error(
        `AI Service failed to choose ring_elimination: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected region order option for a RegionOrderChoice.
   */
  async getRegionOrderChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: RegionOrderChoice['options'],
    requestOptions?: AIServiceRequestOptions
  ): Promise<RegionOrderChoiceResponsePayload> {
    // Cooperative pre-flight cancellation; choice selection is a pure
    // dependency call and should not proceed when canceled.
    requestOptions?.token?.throwIfCanceled(
      'before dispatching AIServiceClient.getRegionOrderChoice'
    );

    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');
    const metrics = getMetricsService();

    try {
      const request: RegionOrderChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI region_order choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<RegionOrderChoiceResponsePayload>(
        '/ai/choice/region_order',
        request
      );

      const duration = performance.now() - startTime;
      metrics.recordAIRequest('success');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, 'success');

      logger.info('AI region_order choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get region_order choice from AI service', {
        playerNumber,
        error,
      });
      const duration = performance.now() - startTime;
      const errorWithType = error as Error & { aiErrorType?: string };
      const aiErrorType = errorWithType.aiErrorType;
      const latencyOutcome: 'success' | 'fallback' | 'timeout' | 'error' =
        aiErrorType === 'timeout' ? 'timeout' : 'error';
      metrics.recordAIRequest('error');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, latencyOutcome);
      if (aiErrorType === 'timeout') {
        metrics.recordAIRequestTimeout();
      }

      throw new Error(
        `AI Service failed to choose region_order: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected line order option for a LineOrderChoice.
   */
  async getLineOrderChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: LineOrderChoice['options'],
    requestOptions?: AIServiceRequestOptions
  ): Promise<LineOrderChoiceResponsePayload> {
    // Cooperative pre-flight cancellation; choice selection is a pure
    // dependency call and should not proceed when canceled.
    requestOptions?.token?.throwIfCanceled('before dispatching AIServiceClient.getLineOrderChoice');

    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');
    const metrics = getMetricsService();

    try {
      const request: LineOrderChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI line_order choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<LineOrderChoiceResponsePayload>(
        '/ai/choice/line_order',
        request
      );

      const duration = performance.now() - startTime;
      metrics.recordAIRequest('success');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, 'success');

      logger.info('AI line_order choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get line_order choice from AI service', {
        playerNumber,
        error,
      });
      const duration = performance.now() - startTime;
      const errorWithType = error as Error & { aiErrorType?: string };
      const aiErrorType = errorWithType.aiErrorType;
      const latencyOutcome: 'success' | 'fallback' | 'timeout' | 'error' =
        aiErrorType === 'timeout' ? 'timeout' : 'error';
      metrics.recordAIRequest('error');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, latencyOutcome);
      if (aiErrorType === 'timeout') {
        metrics.recordAIRequestTimeout();
      }

      throw new Error(
        `AI Service failed to choose line_order: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected capture direction option for a CaptureDirectionChoice.
   */
  async getCaptureDirectionChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: CaptureDirectionChoice['options'],
    requestOptions?: AIServiceRequestOptions
  ): Promise<CaptureDirectionChoiceResponsePayload> {
    // Cooperative pre-flight cancellation; choice selection is a pure
    // dependency call and should not proceed when canceled.
    requestOptions?.token?.throwIfCanceled(
      'before dispatching AIServiceClient.getCaptureDirectionChoice'
    );

    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');
    const metrics = getMetricsService();

    try {
      const request: CaptureDirectionChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI capture_direction choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<CaptureDirectionChoiceResponsePayload>(
        '/ai/choice/capture_direction',
        request
      );

      const duration = performance.now() - startTime;
      metrics.recordAIRequest('success');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, 'success');

      logger.info('AI capture_direction choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get capture_direction choice from AI service', {
        playerNumber,
        error,
      });
      const duration = performance.now() - startTime;
      const errorWithType = error as Error & { aiErrorType?: string };
      const aiErrorType = errorWithType.aiErrorType;
      const latencyOutcome: 'success' | 'fallback' | 'timeout' | 'error' =
        aiErrorType === 'timeout' ? 'timeout' : 'error';
      metrics.recordAIRequest('error');
      metrics.recordAIRequestDuration('python', difficultyLabel, duration / 1000);
      metrics.recordAIRequestLatencyMs(duration, latencyOutcome);
      if (aiErrorType === 'timeout') {
        metrics.recordAIRequestTimeout();
      }

      throw new Error(
        `AI Service failed to choose capture_direction: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Update the ServiceStatusManager with current AI service status.
   * This is called after each AI service interaction.
   */
  private updateServiceStatus(
    status: 'healthy' | 'degraded' | 'unhealthy',
    error?: string,
    latencyMs?: number
  ): void {
    try {
      const statusManager = getServiceStatusManager();
      statusManager.updateServiceStatus('aiService', status, error, latencyMs);
    } catch (e) {
      // Don't fail AI operations if status manager update fails
      logger.debug('Failed to update service status manager', {
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  /**
   * Check if AI service is healthy.
   */
  async healthCheck(): Promise<boolean> {
    const startTime = performance.now();
    try {
      const response = await this.client.get('/health');
      const latencyMs = Math.round(performance.now() - startTime);
      const isHealthy = response.data.status === 'healthy';

      // Update status manager based on health check result
      this.updateServiceStatus(
        isHealthy ? 'healthy' : 'degraded',
        isHealthy ? undefined : 'Health check returned non-healthy status',
        latencyMs
      );

      return isHealthy;
    } catch (error) {
      const latencyMs = Math.round(performance.now() - startTime);
      logger.error('AI Service health check failed', { error });

      // Update status manager on health check failure
      this.updateServiceStatus(
        'unhealthy',
        error instanceof Error ? error.message : 'Health check failed',
        latencyMs
      );

      return false;
    }
  }

  /**
   * Clear AI service cache.
   */
  async clearCache(): Promise<void> {
    try {
      await this.client.delete('/ai/cache');
      logger.info('AI Service cache cleared');
    } catch (error) {
      logger.error('Failed to clear AI cache', { error });
      throw new Error(
        `AI Service failed to clear cache: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get service information.
   */
  async getServiceInfo(): Promise<Record<string, unknown> | null> {
    try {
      const response = await this.client.get('/');
      return response.data;
    } catch (error) {
      logger.error('Failed to get service info', { error });
      return null;
    }
  }

  /**
   * Check if the AI service is currently available (not circuit-broken).
   */
  isServiceAvailable(): boolean {
    return !this.circuitBreaker.isCircuitOpen();
  }
}

// Singleton instance
let aiServiceClient: AIServiceClient | null = null;

/**
 * Get the singleton AI Service client instance
 */
export function getAIServiceClient(): AIServiceClient {
  if (!aiServiceClient) {
    aiServiceClient = new AIServiceClient();
  }
  return aiServiceClient;
}

/**
 * Initialize AI Service client with custom URL
 */
export function initAIServiceClient(baseURL: string): AIServiceClient {
  aiServiceClient = new AIServiceClient(baseURL);
  return aiServiceClient;
}
