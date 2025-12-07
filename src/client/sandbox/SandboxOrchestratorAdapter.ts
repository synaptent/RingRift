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

import type { GameState, Move, GameResult } from '../../shared/engine';
import { hashGameState } from '../../shared/engine';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import {
  processTurn,
  validateMove,
  getValidMoves,
  type ProcessTurnOptions,
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
  /**
   * When true, territory decision (region_order) auto-resolution is skipped.
   * This is used in traceMode/replay contexts where explicit process_territory_region
   * moves should be replayed instead of auto-resolving.
   */
  skipTerritoryAutoResolve?: boolean;
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
  victoryResult?: GameResult | undefined;

  /**
   * Canonical explanation for why the game ended, when available. Derived
   * from the shared orchestrator's VictoryState.gameEndExplanation and
   * threaded through to sandbox hosts for HUD/Victory surfaces.
   */
  gameEndExplanation?: GameEndExplanation | undefined;

  /** Error message if move failed */
  error?: string | undefined;

  /** Whether a decision was requested during processing */
  decisionRequested?: boolean | undefined;

  /** Metadata about the move processing */
  metadata?: SandboxMoveMetadata | undefined;
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
  private readonly callbacks?: SandboxAdapterCallbacks | undefined;
  private readonly skipTerritoryAutoResolve: boolean;

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
    this.skipTerritoryAutoResolve = deps.skipTerritoryAutoResolve ?? false;
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
    const initialState = this.stateAccessor.getGameState();
    const hashBefore = hashGameState(initialState);

    this.callbacks?.debugHook?.('before-processMove', initialState);
    this.callbacks?.onMoveStarted?.(move);

    try {
      // Validate the move first against the current state
      const validation = validateMove(initialState, move);
      if (!validation.valid) {
        return {
          success: false,
          nextState: initialState,
          error: validation.reason || 'Invalid move',
        };
      }

      const delegates = this.createProcessingDelegates();

      // Helper to run processTurn and accumulate phase metadata
      let phasesTraversed: string[] = [];
      const processTurnOptions: ProcessTurnOptions = {
        // In replay/traceMode, don't auto-process single territory regions so
        // explicit process_territory_region moves from the recording are used.
        skipSingleTerritoryAutoProcess: this.skipTerritoryAutoResolve,
        // In replay/traceMode, also avoid auto-processing single-line
        // line_processing phases; instead, surface explicit process_line
        // decisions so recorded move sequences remain the sole authority.
        skipAutoLineProcessing: this.skipTerritoryAutoResolve,
      };
      const runProcessTurn = (state: GameState, moveToApply: Move): ProcessTurnResult => {
        const result = processTurn(state, moveToApply, processTurnOptions);
        if (result.metadata?.phasesTraversed?.length) {
          phasesTraversed = phasesTraversed.concat(result.metadata.phasesTraversed);
        }
        return result;
      };

      // Apply the primary move first so the sandbox engine/state accessor
      // reflect the post-move board BEFORE any pending decisions are surfaced.
      let workingState = initialState;
      let result = runProcessTurn(workingState, move);
      workingState = result.nextState;
      this.stateAccessor.updateGameState(workingState);

      // Resolve any pending decisions (line order, rewards, territory, elimination)
      // in-process, updating the sandbox game state after each canonical move so
      // the UI always sees the latest board snapshot during decision phases.
      while (result.status === 'awaiting_decision' && result.pendingDecision) {
        const decision = result.pendingDecision;

        // Chain-capture decisions are handled specially: expose the available
        // continuation moves via getValidMoves() and return without auto-resolving.
        if (decision.type === 'chain_capture') {
          this.chainCaptureOptions = decision.options;
          break;
        }

        // Territory decisions (region_order) are skipped when skipTerritoryAutoResolve
        // is enabled. This is used in replay/traceMode contexts where explicit
        // process_territory_region moves should come from the recording.
        if (decision.type === 'region_order' && this.skipTerritoryAutoResolve) {
          break;
        }

        // Line-order decisions are also skipped when skipTerritoryAutoResolve
        // is enabled (trace/replay mode). In that context we want explicit
        // process_line / choose_line_reward moves from the recording to drive
        // line processing, rather than auto-selecting a line inside the
        // adapter.
        if (decision.type === 'line_order' && this.skipTerritoryAutoResolve) {
          break;
        }

        // RR-CANON-R076: Handle required no-action decisions from core layer.
        // These are returned when a phase has no available actions.
        // - In replay/traceMode: break and let the caller apply explicit moves
        // - In live play: auto-apply the single no-action option for UX convenience
        if (
          decision.type === 'no_line_action_required' ||
          decision.type === 'no_territory_action_required' ||
          decision.type === 'no_movement_action_required' ||
          decision.type === 'no_placement_action_required'
        ) {
          if (this.skipTerritoryAutoResolve) {
            // Replay/traceMode: require explicit move from recording
            break;
          }
          // Live play: synthesize and auto-apply the required no-action move
          const stateForMove = this.stateAccessor.getGameState();
          const moveNumber = stateForMove.moveHistory.length + 1;

          let moveType: Move['type'];
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
            default:
              // Should be unreachable given the guard above.
              break;
          }

          // If for some reason we did not recognise the decision type, fall
          // back to letting the normal decision flow handle it.
          if (!moveType) {
            break;
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

          result = runProcessTurn(workingState, noActionMove);
          workingState = result.nextState;
          this.stateAccessor.updateGameState(workingState);
          continue;
        }

        // For all other decision types, mirror processTurnAsync semantics:
        // emit a decision_required event, delegate to the decision handler,
        // then emit decision_resolved and continue processing.
        delegates.onProcessingEvent?.({
          type: 'decision_required',
          timestamp: new Date(),
          payload: { decision },
        });

        const chosenMove = await delegates.resolveDecision(decision);

        delegates.onProcessingEvent?.({
          type: 'decision_resolved',
          timestamp: new Date(),
          payload: { decision, chosenMove },
        });

        result = runProcessTurn(workingState, chosenMove);
        workingState = result.nextState;
        this.stateAccessor.updateGameState(workingState);
      }

      // If we did not end in a chain_capture decision, clear any stale options.
      if (
        !(result.status === 'awaiting_decision' && result.pendingDecision?.type === 'chain_capture')
      ) {
        this.chainCaptureOptions = undefined;
      }

      const hashAfter = hashGameState(workingState);
      const durationMs = Date.now() - startTime;

      // Convert victory result if present
      let victoryResult: GameResult | undefined;
      if (result.victoryResult?.isGameOver && result.victoryResult.winner !== undefined) {
        victoryResult = this.convertVictoryResult(result);
      }
      const gameEndExplanation: GameEndExplanation | undefined =
        result.victoryResult?.gameEndExplanation;

      const moveResult: SandboxMoveResult = {
        success: true,
        nextState: workingState,
        victoryResult,
        gameEndExplanation,
        metadata: {
          hashBefore,
          hashAfter,
          stateChanged: hashBefore !== hashAfter,
          durationMs,
          phasesTraversed,
        },
      };

      this.callbacks?.debugHook?.('after-processMove', workingState);
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
        nextState: initialState,
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
      const gameEndExplanation: GameEndExplanation | undefined =
        result.victoryResult?.gameEndExplanation;

      return {
        success: true,
        nextState: result.nextState,
        victoryResult,
        gameEndExplanation,
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
  public previewMove(move: Move): {
    nextState: GameState;
    valid: boolean;
    reason?: string | undefined;
  } {
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

    // Delegate to the core orchestrator for interactive moves.
    const interactiveMoves = getValidMoves(state);

    // Per RR-CANON-R076, the core rules layer no longer fabricates
    // no_*_action bookkeeping moves for placement/movement. For
    // sandbox UX, we preserve the historical behaviour at the
    // adapter/host layer by synthesising those moves when there are
    // no interactive options in the corresponding phase.
    if (interactiveMoves.length === 0 && state.gameStatus === 'active') {
      const moveNumber = state.moveHistory.length + 1;

      if (state.currentPhase === 'ring_placement') {
        const noPlacement: Move = {
          id: `no-placement-action-${moveNumber}`,
          type: 'no_placement_action',
          player: state.currentPlayer,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        };
        return [noPlacement];
      }

      if (state.currentPhase === 'movement') {
        const noMovement: Move = {
          id: `no-movement-action-${moveNumber}`,
          type: 'no_movement_action',
          player: state.currentPlayer,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        };
        return [noMovement];
      }
    }

    return interactiveMoves;
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
