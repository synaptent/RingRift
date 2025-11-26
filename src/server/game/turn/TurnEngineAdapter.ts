/**
 * TurnEngineAdapter - Thin adapter over the shared orchestrator
 *
 * This adapter wraps the canonical processTurn() orchestrator for backend use,
 * delegating all rules logic to the shared engine while handling backend-specific
 * concerns like WebSocket notifications, persistence, and player interactions.
 *
 * Part of Phase 3 Rules Engine Consolidation - see:
 * - docs/drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md
 * - docs/drafts/PHASE3_ADAPTER_MIGRATION_AUDIT.md
 *
 * MIGRATION STATUS: Proof-of-concept implementation
 * This file demonstrates the adapter pattern. Full integration requires:
 * 1. GameEngine.makeMove() to delegate here
 * 2. Wiring to actual WebSocket/interaction handlers
 * 3. History persistence integration
 */

import type { GameState, Move, GameResult, Position } from '../../../shared/types/game';
import type {
  ProcessTurnResult,
  TurnProcessingDelegates,
  PendingDecision,
  VictoryState,
} from '../../../shared/engine/orchestration/types';
import {
  processTurnAsync,
  validateMove,
  getValidMoves,
  hasValidMoves,
} from '../../../shared/engine/orchestration/turnOrchestrator';

// ═══════════════════════════════════════════════════════════════════════════
// INTERFACES - Abstract dependencies for testability
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Interface for accessing and updating game state.
 * Adapters implement this to abstract away the specific session/context.
 */
export interface StateAccessor {
  /** Get current game state */
  getGameState(): GameState;
  /** Update game state after processing */
  updateGameState(state: GameState): void;
  /** Get player info by number */
  getPlayerInfo(playerNumber: number): { type: 'human' | 'ai' } | undefined;
}

/**
 * Interface for handling player decisions.
 * Backend implements this to route to WebSocket, sandbox implements for local UI.
 */
export interface DecisionHandler {
  /**
   * Request a decision from a player.
   * @param decision The pending decision with options
   * @returns The selected Move
   */
  requestDecision(decision: PendingDecision): Promise<Move>;
}

/**
 * Interface for emitting game events.
 */
export interface EventEmitter {
  emit(event: string, payload: unknown): void;
}

/**
 * Combined dependencies for the adapter.
 */
export interface TurnEngineAdapterDeps {
  stateAccessor: StateAccessor;
  decisionHandler: DecisionHandler;
  eventEmitter?: EventEmitter;
  debugHook?: (label: string, state: GameState) => void;
}

/**
 * Result of processing a move via the adapter.
 */
export interface AdapterMoveResult {
  success: boolean;
  nextState: GameState;
  victoryResult: GameResult | undefined;
  error: string | undefined;
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTER IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * TurnEngineAdapter wraps the shared orchestrator for backend game processing.
 *
 * Key responsibilities:
 * - Convert WebSocket move commands to orchestrator Move format
 * - Handle PendingDecision responses (prompt player via WebSocket)
 * - Persist state changes after orchestrator returns
 * - Emit game events for spectators
 *
 * NOT responsible for:
 * - Rules logic (delegated to orchestrator)
 * - Victory detection (delegated to orchestrator)
 * - Move validation (delegated to orchestrator)
 */
export class TurnEngineAdapter {
  private readonly deps: TurnEngineAdapterDeps;

  constructor(deps: TurnEngineAdapterDeps) {
    this.deps = deps;
  }

  /**
   * Process a move using the canonical orchestrator.
   *
   * This is the primary entry point that GameEngine.makeMove() should delegate to
   * after Phase 3 migration is complete.
   */
  async processMove(move: Move): Promise<AdapterMoveResult> {
    const { stateAccessor, decisionHandler, eventEmitter, debugHook } = this.deps;
    const beforeState = stateAccessor.getGameState();

    // Debug checkpoint before processing
    debugHook?.('before-processMove', beforeState);

    // Create delegates that route decisions to players
    const delegates: TurnProcessingDelegates = {
      resolveDecision: async (decision: PendingDecision): Promise<Move> => {
        const playerInfo = stateAccessor.getPlayerInfo(decision.player);

        // For AI players, auto-select first option
        if (playerInfo?.type === 'ai') {
          return this.autoSelectForAI(decision);
        }

        // For human players, delegate to handler
        return decisionHandler.requestDecision(decision);
      },
      onProcessingEvent: (event) => {
        eventEmitter?.emit('game:processing_event', event);
      },
    };

    try {
      // Delegate to canonical orchestrator
      const result = await processTurnAsync(beforeState, move, delegates);

      // Update state
      stateAccessor.updateGameState(result.nextState);

      // Debug checkpoint after processing
      debugHook?.('after-processMove', result.nextState);

      // Emit state update
      eventEmitter?.emit('game:state_update', {
        state: result.nextState,
        move,
        metadata: result.metadata,
      });

      // Handle victory if game ended
      if (result.victoryResult?.isGameOver) {
        const gameResult = this.convertVictoryToGameResult(result.victoryResult);
        eventEmitter?.emit('game:ended', { result: gameResult });

        return {
          success: true,
          nextState: result.nextState,
          victoryResult: gameResult,
          error: undefined,
        };
      }

      return {
        success: true,
        nextState: result.nextState,
        victoryResult: undefined,
        error: undefined,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        nextState: beforeState,
        victoryResult: undefined,
        error: errorMessage,
      };
    }
  }

  /**
   * Validate a move without applying it.
   */
  validateMoveOnly(state: GameState, move: Move): { valid: boolean; reason: string | undefined } {
    const result = validateMove(state, move);
    return {
      valid: result.valid,
      reason: result.reason ?? undefined,
    };
  }

  /**
   * Get all valid moves for the current player.
   */
  getValidMovesFor(state: GameState): Move[] {
    return getValidMoves(state);
  }

  /**
   * Check if any valid moves exist for the current player.
   */
  hasAnyValidMoves(state: GameState): boolean {
    return hasValidMoves(state);
  }

  /**
   * Auto-select a decision option for AI players.
   */
  private autoSelectForAI(decision: PendingDecision): Move {
    if (decision.options.length === 0) {
      throw new Error(`No options available for AI decision: ${decision.type}`);
    }
    // Simple strategy: pick first option
    return decision.options[0];
  }

  /**
   * Convert orchestrator VictoryState to backend GameResult format.
   */
  private convertVictoryToGameResult(victory: VictoryState): GameResult | undefined {
    if (!victory || !victory.isGameOver) {
      return undefined;
    }

    const ringsEliminated: Record<number, number> = {};
    const territorySpaces: Record<number, number> = {};
    const ringsRemaining: Record<number, number> = {};

    for (const score of victory.scores) {
      ringsEliminated[score.player] = score.eliminatedRings;
      territorySpaces[score.player] = score.territorySpaces;
      ringsRemaining[score.player] = score.ringsOnBoard + score.ringsInHand;
    }

    // Map orchestrator reasons to game result reasons
    const reasonMap: Record<string, GameResult['reason']> = {
      ring_elimination: 'ring_elimination',
      territory_control: 'territory_control',
      last_player_standing: 'last_player_standing',
      stalemate_resolution: 'draw',
      resignation: 'resignation',
    };

    // GameResult.winner must be a number (not undefined)
    // If no winner, we return undefined for the whole result
    if (victory.winner === undefined) {
      return undefined;
    }

    return {
      winner: victory.winner,
      reason: reasonMap[victory.reason || ''] || 'game_completed',
      finalScore: {
        ringsEliminated,
        territorySpaces,
        ringsRemaining,
      },
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FACTORY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a TurnEngineAdapter with a simple state holder.
 * Useful for testing or simple integrations.
 */
export function createSimpleAdapter(
  initialState: GameState,
  decisionHandler: DecisionHandler
): {
  adapter: TurnEngineAdapter;
  getState: () => GameState;
} {
  let currentState = initialState;

  const stateAccessor: StateAccessor = {
    getGameState: () => currentState,
    updateGameState: (state) => {
      currentState = state;
    },
    getPlayerInfo: (playerNumber) => {
      const player = currentState.players.find((p) => p.playerNumber === playerNumber);
      return player ? { type: player.type } : undefined;
    },
  };

  const adapter = new TurnEngineAdapter({
    stateAccessor,
    decisionHandler,
  });

  return {
    adapter,
    getState: () => currentState,
  };
}

/**
 * Create a mock decision handler that auto-selects first option.
 * Useful for testing.
 */
export function createAutoSelectDecisionHandler(): DecisionHandler {
  return {
    requestDecision: async (decision: PendingDecision): Promise<Move> => {
      if (decision.options.length === 0) {
        throw new Error(`No options for decision: ${decision.type}`);
      }
      return decision.options[0];
    },
  };
}
