/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Sandbox Orchestrator Adapter
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Adapter that wraps the shared orchestrator's processTurn() for client
 * sandbox use. This provides the same interface as the backend
 * TurnEngineAdapter but tailored for browser/local game contexts.
 *
 * Key differences from backend adapter:
 * - No persistence layer (state is managed locally)
 * - No WebSocket notifications (UI updates are synchronous)
 * - Supports "preview" mode for what-if analysis
 */

import type { GameState, Move, GameResult, Position } from '../../shared/engine';
import { hashGameState } from '../../shared/engine';
import {
  processTurn,
  processTurnAsync,
  validateMove,
  getValidMoves,
} from '../../shared/engine/orchestration/turnOrchestrator';
import type {
  ProcessTurnResult,
  PendingDecision,
  TurnProcessingDelegates,
} from '../../shared/engine/orchestration/types';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * State accessor for the sandbox adapter.
 * Provides read/write access to the local game state.
 */
export interface SandboxStateAccessor {
  /** Get a defensive copy of the current game state */
  getGameState(): GameState;

  /** Update the game state with a new state */
  updateGameState(newState: GameState): void;

  /** Get player info for decision handling */
  getPlayerInfo(playerId: string): {
    type: 'human' | 'ai';
    aiDifficulty?: number;
  };
}

/**
 * Decision handler for the sandbox adapter.
 * Handles user decisions when the orchestrator encounters choices.
 */
export interface SandboxDecisionHandler {
  /**
   * Request a decision from the player (human or AI).
   * For humans, this typically shows a dialog and waits for selection.
   * For AI, this automatically selects based on heuristics.
   */
  requestDecision(decision: PendingDecision): Promise<Move>;
}

/**
 * Optional callbacks for sandbox-specific features.
 */
export interface SandboxAdapterCallbacks {
  /** Called before applying a move (for preview/animation) */
  onMoveStarted?: (move: Move) => void;

  /** Called after a move is applied successfully */
  onMoveCompleted?: (move: Move, result: SandboxMoveResult) => void;

  /** Called when a decision is required */
  onDecisionRequired?: (decision: PendingDecision) => void;

  /** Called when an error occurs */
  onError?: (error: Error, context: string) => void;

  /** Debug hook for development/testing */
  debugHook?: (label: string, state: GameState) => void;
}

/**
 * Combined dependencies for the sandbox adapter.
 */
export interface SandboxAdapterDeps {
  stateAccessor: SandboxStateAccessor;
  decisionHandler: SandboxDecisionHandler;
  callbacks?: SandboxAdapterCallbacks;
}

/**
 * Result of processing a move through the sandbox adapter.
 */
export interface SandboxMoveResult {
  /** Whether the move was applied successfully */
  success: boolean;

  /** The resulting game state after move application */
  nextState: GameState;

  /** Victory result if the game ended */
  victoryResult?: GameResult;

  /** Error message if move failed */
  error?: string;

  /** Whether a decision was requested during processing */
  decisionRequested?: boolean;

  /** Metadata about the move processing */
  metadata?: SandboxMoveMetadata;
}

/**
 * Metadata about move processing.
 */
export interface SandboxMoveMetadata {
  /** State hash before the move */
  hashBefore: string;
  /** State hash after the move */
  hashAfter: string;
  /** Whether the state changed */
  stateChanged: boolean;
  /** Processing duration in milliseconds */
  durationMs: number;
  /** Phases traversed during processing */
  phasesTraversed: string[];
}

// ═══════════════════════════════════════════════════════════════════════════
// Adapter Implementation
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Adapter that wraps the shared orchestrator for sandbox use.
 *
 * Usage:
 * ```typescript
 * const adapter = new SandboxOrchestratorAdapter({
 *   stateAccessor: {
 *     getGameState: () => engine.getGameState(),
 *     updateGameState: (state) => engine.setGameState(state),
 *     getPlayerInfo: (id) => ({ type: 'human' }),
 *   },
 *   decisionHandler: {
 *     requestDecision: async (decision) => {
 *       // Show dialog and wait for user selection
 *       return selectedMove;
 *     },
 *   },
 * });
 *
 * const result = await adapter.processMove(move);
 * if (result.success) {
 *   // State automatically updated
 * }
 * ```
 */
export class SandboxOrchestratorAdapter {
  private readonly stateAccessor: SandboxStateAccessor;
  private readonly decisionHandler: SandboxDecisionHandler;
  private readonly callbacks?: SandboxAdapterCallbacks;

  /**
   * Chain capture continuation moves, populated when processTurnAsync returns
   * with a chain_capture pending decision. These are returned by getValidMoves()
   * when the phase is chain_capture.
   */
  private chainCaptureOptions: Move[] | undefined;

  constructor(deps: SandboxAdapterDeps) {
    this.stateAccessor = deps.stateAccessor;
    this.decisionHandler = deps.decisionHandler;
    this.callbacks = deps.callbacks;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Main API
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Process a move through the orchestrator.
   *
   * This is the main entry point for turn processing. It:
   * 1. Validates the move
   * 2. Applies the move via orchestrator
   * 3. Handles any decisions that arise
   * 4. Updates the game state
   * 5. Returns the result
   */
  public async processMove(move: Move): Promise<SandboxMoveResult> {
    const startTime = Date.now();
    const state = this.stateAccessor.getGameState();
    const hashBefore = hashGameState(state);

    this.callbacks?.debugHook?.('before-processMove', state);
    this.callbacks?.onMoveStarted?.(move);

    try {
      // Validate the move first
      const validation = validateMove(state, move);
      if (!validation.valid) {
        return {
          success: false,
          nextState: state,
          error: validation.reason || 'Invalid move',
        };
      }

      // Process through the orchestrator
      const delegates = this.createProcessingDelegates();
      const result = await processTurnAsync(state, move, delegates);

      // Update state
      this.stateAccessor.updateGameState(result.nextState);

      // Handle chain capture pending decision: store the continuation options
      // so getValidMoves() can return them during chain_capture phase.
      if (
        result.status === 'awaiting_decision' &&
        result.pendingDecision?.type === 'chain_capture'
      ) {
        this.chainCaptureOptions = result.pendingDecision.options;
      } else {
        // Clear chain capture options when not in chain capture
        this.chainCaptureOptions = undefined;
      }

      const hashAfter = hashGameState(result.nextState);
      const durationMs = Date.now() - startTime;

      // Convert victory result if present
      let victoryResult: GameResult | undefined;
      if (result.victoryResult?.isGameOver && result.victoryResult.winner !== undefined) {
        victoryResult = this.convertVictoryResult(result);
      }

      const moveResult: SandboxMoveResult = {
        success: true,
        nextState: result.nextState,
        victoryResult,
        metadata: {
          hashBefore,
          hashAfter,
          stateChanged: hashBefore !== hashAfter,
          durationMs,
          phasesTraversed: result.metadata?.phasesTraversed ?? [],
        },
      };

      this.callbacks?.debugHook?.('after-processMove', result.nextState);
      this.callbacks?.onMoveCompleted?.(move, moveResult);

      return moveResult;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.callbacks?.onError?.(
        error instanceof Error ? error : new Error(errorMessage),
        'processMove'
      );

      return {
        success: false,
        nextState: state,
        error: errorMessage,
      };
    }
  }

  /**
   * Process a move synchronously (no decision handling).
   *
   * This is useful for:
   * - AI auto-play where decisions are made automatically
   * - Preview/what-if analysis
   * - Single-outcome moves that don't require choices
   *
   * If a decision is required, returns with decisionRequested=true
   * and the pendingDecision in the result.
   */
  public processMoveSync(move: Move): SandboxMoveResult & { pendingDecision?: PendingDecision } {
    const startTime = Date.now();
    const state = this.stateAccessor.getGameState();
    const hashBefore = hashGameState(state);

    try {
      const validation = validateMove(state, move);
      if (!validation.valid) {
        return {
          success: false,
          nextState: state,
          error: validation.reason || 'Invalid move',
        };
      }

      const result = processTurn(state, move);

      // If a decision is required, don't update state
      if (result.status === 'awaiting_decision' && result.pendingDecision) {
        return {
          success: false,
          nextState: state,
          decisionRequested: true,
          pendingDecision: result.pendingDecision,
        };
      }

      // Update state
      this.stateAccessor.updateGameState(result.nextState);

      const hashAfter = hashGameState(result.nextState);
      const durationMs = Date.now() - startTime;

      let victoryResult: GameResult | undefined;
      if (result.victoryResult?.isGameOver && result.victoryResult.winner !== undefined) {
        victoryResult = this.convertVictoryResult(result);
      }

      return {
        success: true,
        nextState: result.nextState,
        victoryResult,
        metadata: {
          hashBefore,
          hashAfter,
          stateChanged: hashBefore !== hashAfter,
          durationMs,
          phasesTraversed: result.metadata?.phasesTraversed ?? [],
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        nextState: state,
        error: errorMessage,
      };
    }
  }

  /**
   * Preview a move without applying it to the actual state.
   *
   * This allows for what-if analysis without modifying game state.
   * Returns the resulting state that would occur if the move was applied.
   */
  public previewMove(move: Move): { nextState: GameState; valid: boolean; reason?: string } {
    const state = this.stateAccessor.getGameState();

    const validation = validateMove(state, move);
    if (!validation.valid) {
      return {
        nextState: state,
        valid: false,
        reason: validation.reason,
      };
    }

    // Use synchronous processing for preview (no decision handling)
    const result = processTurn(state, move);

    return {
      nextState: result.nextState,
      valid: true,
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Validation & Enumeration (delegated to orchestrator)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Validate a move without applying it.
   */
  public validateMove(move: Move): { valid: boolean; reason?: string } {
    const state = this.stateAccessor.getGameState();
    return validateMove(state, move);
  }

  /**
   * Get all valid moves for the current player and phase.
   *
   * During chain_capture phase, returns the stored continuation options
   * from the previous capture move's pending decision.
   */
  public getValidMoves(): Move[] {
    const state = this.stateAccessor.getGameState();

    // During chain_capture phase, return stored continuation options
    if (state.currentPhase === 'chain_capture' && this.chainCaptureOptions) {
      return this.chainCaptureOptions;
    }

    return getValidMoves(state);
  }

  /**
   * Check if a specific move is in the valid moves list.
   */
  public isMoveValid(move: Move): boolean {
    const valid = this.validateMove(move);
    return valid.valid;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // State Queries
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Get the current game state.
   */
  public getGameState(): GameState {
    return this.stateAccessor.getGameState();
  }

  /**
   * Check if the game is in a terminal state.
   */
  public isGameOver(): boolean {
    const state = this.stateAccessor.getGameState();
    return state.gameStatus !== 'active';
  }

  /**
   * Get the current player number.
   */
  public getCurrentPlayer(): number {
    return this.stateAccessor.getGameState().currentPlayer;
  }

  /**
   * Get the current game phase.
   */
  public getCurrentPhase(): string {
    return this.stateAccessor.getGameState().currentPhase;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private Helpers
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Create the TurnProcessingDelegates for async processing.
   */
  private createProcessingDelegates(): TurnProcessingDelegates {
    return {
      resolveDecision: async (decision: PendingDecision): Promise<Move> => {
        this.callbacks?.onDecisionRequired?.(decision);

        // Check if this is an AI player
        const currentPlayer = this.stateAccessor.getGameState().currentPlayer;
        const playerId = `player-${currentPlayer}`;
        const playerInfo = this.stateAccessor.getPlayerInfo(playerId);

        if (playerInfo.type === 'ai') {
          // AI auto-selects from options
          return this.autoSelectDecision(decision);
        }

        // Human player - delegate to handler
        return this.decisionHandler.requestDecision(decision);
      },
      onProcessingEvent: (event) => {
        // Forward events to debug hook if provided
        if (event.type === 'decision_required' && event.payload?.decision) {
          this.callbacks?.debugHook?.('decision-required', this.stateAccessor.getGameState());
        }
      },
    };
  }

  /**
   * Auto-select a decision option for AI players.
   * Uses simple heuristics - for more sophisticated AI behavior,
   * the decisionHandler should implement custom logic.
   */
  private autoSelectDecision(decision: PendingDecision): Move {
    // Default to first option
    if (decision.options.length > 0) {
      return decision.options[0];
    }

    // Fallback: create a minimal skip move
    const state = this.stateAccessor.getGameState();
    return {
      id: `auto-decision-${Date.now()}`,
      type: 'skip_placement',
      player: decision.player,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: state.moveHistory.length + 1,
    };
  }

  /**
   * Convert orchestrator VictoryState to sandbox GameResult.
   */
  private convertVictoryResult(result: ProcessTurnResult): GameResult | undefined {
    if (!result.victoryResult?.isGameOver || result.victoryResult.winner === undefined) {
      return undefined;
    }

    const state = result.nextState;
    const ringsEliminated: { [playerNumber: number]: number } = {};
    const territorySpaces: { [playerNumber: number]: number } = {};
    const ringsRemaining: { [playerNumber: number]: number } = {};

    for (const player of state.players) {
      ringsEliminated[player.playerNumber] = player.eliminatedRings;
      territorySpaces[player.playerNumber] = player.territorySpaces;
      ringsRemaining[player.playerNumber] = 0;
    }

    for (const stack of state.board.stacks.values()) {
      const owner = stack.controllingPlayer;
      ringsRemaining[owner] = (ringsRemaining[owner] || 0) + stack.stackHeight;
    }

    // Map orchestrator victory reason to GameResult reason
    const reasonMap: { [key: string]: GameResult['reason'] } = {
      ring_elimination: 'ring_elimination',
      territory_control: 'territory_control',
      last_player_standing: 'last_player_standing',
      stalemate_resolution: 'game_completed',
      resignation: 'resignation',
      timeout: 'timeout',
      draw: 'draw',
      abandonment: 'abandonment',
    };

    const rawReason = result.victoryResult.reason || 'ring_elimination';
    const reason: GameResult['reason'] = reasonMap[rawReason] || 'game_completed';

    return {
      winner: result.victoryResult.winner,
      reason,
      finalScore: {
        ringsEliminated,
        territorySpaces,
        ringsRemaining,
      },
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Factory Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a sandbox adapter with minimal configuration.
 *
 * This is a convenience function for simple use cases where
 * you just need basic state management.
 */
export function createSandboxAdapter(
  getState: () => GameState,
  setState: (state: GameState) => void,
  decisionHandler: SandboxDecisionHandler
): SandboxOrchestratorAdapter {
  return new SandboxOrchestratorAdapter({
    stateAccessor: {
      getGameState: getState,
      updateGameState: setState,
      getPlayerInfo: () => ({ type: 'human' }),
    },
    decisionHandler,
  });
}

/**
 * Create a sandbox adapter configured for AI-only games.
 *
 * All decisions are auto-resolved using simple heuristics.
 */
export function createAISandboxAdapter(
  getState: () => GameState,
  setState: (state: GameState) => void
): SandboxOrchestratorAdapter {
  return new SandboxOrchestratorAdapter({
    stateAccessor: {
      getGameState: getState,
      updateGameState: setState,
      getPlayerInfo: () => ({ type: 'ai' }),
    },
    decisionHandler: {
      requestDecision: async (decision) => {
        // Return first option for AI
        if (decision.options.length > 0) {
          return decision.options[0];
        }
        throw new Error('No decision options available');
      },
    },
  });
}
