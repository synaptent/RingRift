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
 * MIGRATION STATUS: Phase A – backend orchestrator path live
 *
 * - In non-test environments, GameEngine.makeMove() now delegates to this
 *   adapter by default (see GameEngine.processMoveViaAdapter and
 *   config.isTest / useOrchestratorAdapter).
 * - Decision handling is currently test-only (auto-select or throw); the
 *   production WebSocket interaction layer will wire a real DecisionHandler
 *   for interactive phases in a later phase.
 * - Event emission is intentionally minimal (`game:state_update`,
 *   `game:ended`) and can be extended to drive richer spectator updates and
 *   history persistence once rollout is fully complete.
 */

import type { GameState, Move, GameResult } from '../../../shared/types/game';
import type {
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
import { flagEnabled } from '../../../shared/utils/envFlags';

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
  /**
   * When true, the adapter operates in replay mode:
   * - Auto-processing of single-option decisions (lines, territory) is disabled.
   * - The decision loop is broken immediately when a decision is required,
   *   returning success: true but with the state left in the decision phase.
   *   This allows the replay driver to supply the explicit decision move from
   *   the recording.
   */
  replayMode?: boolean;
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
    const { stateAccessor, decisionHandler, eventEmitter, debugHook, replayMode } = this.deps;
    const beforeState = stateAccessor.getGameState();

    // Debug checkpoint before processing
    debugHook?.('before-processMove', beforeState);

    // Create delegates that route decisions to players
    const delegates: TurnProcessingDelegates = {
      resolveDecision: async (decision: PendingDecision): Promise<Move> => {
        // In replay mode, we must NOT resolve decisions via delegates. The
        // replay driver will supply the explicit decision move from the
        // recording in the next call. We throw here to ensure the orchestrator
        // loop breaks (though processTurnAsync should handle this if we pass
        // the right options).
        if (replayMode) {
          throw new Error(
            `[TurnEngineAdapter] resolveDecision called in replayMode for ${decision.type} - this should not happen if processTurnAsync is configured correctly`
          );
        }

        // Core may surface required no-action decisions when a phase has no
        // interactive moves (RR-CANON-R075/R076). These are non-interactive
        // bookkeeping steps; hosts are responsible for constructing the
        // corresponding no_*_action Move and applying it via the normal API.
        if (
          decision.type === 'no_line_action_required' ||
          decision.type === 'no_territory_action_required' ||
          decision.type === 'no_movement_action_required' ||
          decision.type === 'no_placement_action_required'
        ) {
          const currentState = stateAccessor.getGameState();
          const moveNumber = currentState.moveHistory.length + 1;

          let moveType: Move['type'] | undefined;
          switch (decision.type) {
            case 'no_line_action_required':
              moveType = 'no_line_action';
              break;
            case 'no_territory_action_required':
              moveType = 'no_territory_action';
              break;
            case 'no_movement_action_required':
              moveType = 'no_movement_action';
              break;
            case 'no_placement_action_required':
              moveType = 'no_placement_action';
              break;
          }

          if (!moveType) {
            throw new Error(`Unhandled no-action decision type: ${decision.type}`);
          }

          const noActionMove: Move = {
            id: `auto-${moveType}-${moveNumber}`,
            type: moveType,
            player: decision.player,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          };
          return noActionMove;
        }

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
      // In replay mode, rely on explicit decision moves from the recording by
      // breaking when a decision is required, but otherwise allow normal
      // auto-processing semantics to run.
      const processTurnOptions = replayMode
        ? {
            breakOnDecisionRequired: true,
          }
        : {};
      const result = await processTurnAsync(beforeState, move, delegates, processTurnOptions);

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

      const TRACE_DEBUG_ENABLED = flagEnabled('RINGRIFT_TRACE_DEBUG');

      if (TRACE_DEBUG_ENABLED) {
        const isDecisionHandlerError = errorMessage.includes(
          'DecisionHandler.requestDecision called for'
        );
        const isEliminationTargetDecisionError = errorMessage.includes('elimination_target');

        console.error('[TurnEngineAdapter.processMove] orchestrator error', {
          moveType: move.type,
          player: move.player,
          from: move.from,
          to: move.to,
          currentPhase: beforeState.currentPhase,
          currentPlayer: beforeState.currentPlayer,
          gameStatus: beforeState.gameStatus,
          errorMessage,
          isDecisionHandlerError,
          isEliminationTargetDecisionError,
        });
      }

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
    const interactiveMoves = getValidMoves(state);

    // Per RR-CANON-R076 the core orchestrator does not fabricate
    // no_*_action bookkeeping moves. For backend hosts we preserve
    // the historical behaviour by synthesising the required
    // no_placement_action / no_movement_action moves when there are
    // no interactive options in those phases, so canonical history
    // still records every visited phase.
    if (interactiveMoves.length === 0 && state.gameStatus === 'active') {
      const moveNumber = state.moveHistory.length + 1;

      if (state.currentPhase === 'ring_placement') {
        return [
          {
            id: `no-placement-action-${moveNumber}`,
            type: 'no_placement_action',
            player: state.currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move,
        ];
      }

      if (state.currentPhase === 'movement') {
        return [
          {
            id: `no-movement-action-${moveNumber}`,
            type: 'no_movement_action',
            player: state.currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move,
        ];
      }
    }

    return interactiveMoves;
  }

  /**
   * Check if any valid moves exist for the current player.
   */
  hasAnyValidMoves(state: GameState): boolean {
    return hasValidMoves(state);
  }

  /**
   * Auto-select a decision option for AI players.
   *
   * For elimination_target decisions with no options (rare edge case where
   * orchestrator signals elimination but no eligible stacks exist), we return
   * a no-op elimination move to avoid crashing the game. This mirrors the
   * defensive handling in GameEngine.DecisionHandler.
   */
  private autoSelectForAI(decision: PendingDecision): Move {
    if (decision.options.length === 0) {
      // Defensive handling for elimination_target with no options
      // (mirrors GameEngine.DecisionHandler behavior)
      if (decision.type === 'elimination_target') {
        const noopMove: Move = {
          id: `noop-eliminate-${Date.now()}`,
          type: 'eliminate_rings_from_stack',
          player: decision.player,
          // Use sentinel coordinate; elimination handler treats this as no-op
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 0,
        };
        return noopMove;
      }
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
