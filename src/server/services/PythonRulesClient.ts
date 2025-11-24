import axios, { AxiosInstance } from 'axios';
import { GameState, Move } from '../../shared/types/game';
import { logger } from '../utils/logger';
import { config } from '../config';

export interface RulesEvalResponse {
  valid: boolean;
  // With exactOptionalPropertyTypes enabled, explicitly include undefined
  // in the type of optional properties that may be omitted.
  validationError?: string | undefined;
  nextState?: GameState | undefined;
  stateHash?: string | undefined;
  sInvariant?: number | undefined;
  gameStatus?: GameState['gameStatus'] | undefined;
}

interface RulesEvalResponseWire {
  valid: boolean;
  validation_error?: string;
  next_state?: GameState;
  state_hash?: string;
  s_invariant?: number;
  game_status?: GameState['gameStatus'];
}

/**
 * Lightweight HTTP client for the Python rules engine.
 *
 * Talks to the FastAPI endpoint /rules/evaluate_move exposed by the
 * ai-service and normalises its snake_case response fields into the
 * camelCase RulesEvalResponse shape used by the backend rules façade.
 */
export class PythonRulesClient {
  private readonly client: AxiosInstance;

  constructor(baseURL?: string) {
    const url = baseURL || config.aiService.url;

    this.client = axios.create({
      baseURL: url,
      // Use a bounded per-request timeout for rules evaluations so that
      // slow or unavailable Python backends do not block move application.
      timeout: config.aiService.rulesTimeoutMs,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  async evaluateMove(state: GameState, move: Move): Promise<RulesEvalResponse> {
    try {
      const response = await this.client.post<RulesEvalResponseWire>('/rules/evaluate_move', {
        game_state: state,
        move,
      });

      const data = response.data;

      return {
        valid: data.valid,
        validationError: data.validation_error,
        nextState: data.next_state,
        stateHash: data.state_hash,
        sInvariant: data.s_invariant,
        gameStatus: data.game_status,
      };
    } catch (error) {
      logger.error('Python rules evaluate_move failed', {
        message: (error as any)?.message,
        response: (error as any)?.response?.data,
        status: (error as any)?.response?.status,
      });

      throw error;
    }
  }
}
