/**
 * @fileoverview Sandbox Orchestrator Adapter - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It wraps the shared orchestrator's `processTurn()` for client sandbox use.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - FSM validation: `src/shared/engine/fsm/FSMAdapter.ts`
 * - Types: `src/shared/types/game.ts`
 *
 * This adapter:
 * - Provides the same interface as the backend TurnEngineAdapter
 * - Manages local state (no persistence layer)
 * - Handles decision delegation to UI or AI
 * - Supports "preview" mode for what-if analysis
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 * Per RR-CANON-R070, FSM validation is the canonical move validator.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Implementation Notes
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Key differences from backend adapter:
 * - No persistence layer (state is managed locally)
 * - No WebSocket notifications (UI updates are synchronous)
 * - Supports "preview" mode for what-if analysis
 */

import type { GameState, Move, GameResult, Position } from '../../shared/engine';
import { hashGameState } from '../../shared/engine';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import { createHistoryEntry } from '../../shared/engine/historyHelpers';
import { serializeGameState } from '../../shared/engine/contracts/serialization';
import {
  processTurn,
  getValidMoves,
  type ProcessTurnOptions,
} from '../../shared/engine/orchestration/turnOrchestrator';
import { validateMoveWithFSM } from '../../shared/engine/fsm/FSMAdapter';
import { validatePlacement } from '../../shared/engine/aggregates/PlacementAggregate';
import { normalizeLegacyMove } from '../../shared/engine/legacy/legacyMoveTypes';
import { getSandboxAIServiceAvailable } from '../utils/aiServiceAvailability';
import {
  debugLog,
  isSandboxAiStallDiagnosticsEnabled,
  isTestEnvironment,
} from '../../shared/utils/envFlags';
import { recordSandboxAiDiagnostics } from './sandboxAiDiagnostics';
import type {
  ProcessTurnResult,
  PendingDecision,
  TurnProcessingDelegates,
} from '../../shared/engine/orchestration/types';

const SANDBOX_ORCHESTRATOR_TRACE_DEBUG = isSandboxAiStallDiagnosticsEnabled();

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
   * This is used in traceMode/replay contexts where explicit choose_territory_option
   * moves should be replayed instead of auto-resolving (legacy alias: process_territory_region).
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
      const validation = validateMoveWithFSM(initialState, move);
      if (!validation.valid) {
        return {
          success: false,
          nextState: initialState,
          error: validation.reason || 'Invalid move',
        };
      }

      // For place_ring moves, perform additional board-level validation
      // including no-dead-placement check per RR-CANON-R081/R082.
      // The FSM validation only checks phase/event compatibility, not
      // board-level constraints like whether the placement would result
      // in a stack with no legal moves.
      if (move.type === 'place_ring' && move.to) {
        const placementAction = {
          type: 'PLACE_RING' as const,
          playerId: move.player,
          position: move.to,
          count: move.placementCount ?? 1,
        };
        const placementValidation = validatePlacement(initialState, placementAction);
        if (!placementValidation.valid) {
          return {
            success: false,
            nextState: initialState,
            error:
              placementValidation.reason ||
              `Invalid placement position at (${move.to.x}, ${move.to.y})`,
          };
        }
      }

      const delegates = this.createProcessingDelegates();
      const moveToApply = normalizeLegacyMove(move);

      // Helper to run processTurn and accumulate phase metadata
      let phasesTraversed: string[] = [];
      const processTurnOptions: ProcessTurnOptions = {
        // Sandbox is intentionally replay-tolerant to support traceMode,
        // legacy recordings, and AI-driven multi-player flows.
        replayCompatibility: true,
        // In replay/traceMode, don't auto-process single territory regions so
        // explicit choose_territory_option moves from the recording are used
        // (legacy alias: process_territory_region).
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
      let result = runProcessTurn(workingState, moveToApply);

      // RR-DEBUG-2025-12-13: Trace decision loop entry for all moves
      debugLog(
        SANDBOX_ORCHESTRATOR_TRACE_DEBUG,
        '[SandboxOrchestratorAdapter.processMove] After runProcessTurn:',
        {
          moveType: moveToApply.type,
          resultStatus: result.status,
          pendingDecision: result.pendingDecision?.type ?? 'none',
          nextPhase: result.nextState.currentPhase,
          nextPlayer: result.nextState.currentPlayer,
        }
      );

      const afterPrimary = result.nextState;
      const primaryEntry = createHistoryEntry(workingState, afterPrimary, moveToApply, {
        normalizeMoveNumber: true,
      });
      workingState = {
        ...afterPrimary,
        moveHistory: [
          ...workingState.moveHistory,
          { ...moveToApply, moveNumber: primaryEntry.moveNumber },
        ],
        history: [...workingState.history, primaryEntry],
        lastMoveAt: new Date(),
      };
      this.stateAccessor.updateGameState(workingState);

      // Resolve any pending decisions (line order, rewards, territory, elimination)
      // in-process, updating the sandbox game state after each canonical move so
      // the UI always sees the latest board snapshot during decision phases.

      // DEBUG: Log decision loop entry conditions for chain capture debugging
      if (
        workingState.currentPhase === 'chain_capture' ||
        result.pendingDecision?.type === 'chain_capture'
      ) {
        // eslint-disable-next-line no-console
        console.log('[SandboxOrchestratorAdapter.processMove] Chain capture scenario detected:', {
          resultStatus: result.status,
          hasPendingDecision: !!result.pendingDecision,
          pendingDecisionType: result.pendingDecision?.type,
          pendingDecisionOptionsLength: result.pendingDecision?.options?.length ?? 0,
          workingStatePhase: workingState.currentPhase,
          chainCapturePosition: workingState.chainCapturePosition,
        });
      }

      // RR-DEBUG-2025-12-13: Trace decision loop iterations
      let decisionLoopIteration = 0;
      while (result.status === 'awaiting_decision' && result.pendingDecision) {
        const decision = result.pendingDecision;
        decisionLoopIteration++;

        // RR-DEBUG-2025-12-13: Log every decision loop iteration
        debugLog(
          SANDBOX_ORCHESTRATOR_TRACE_DEBUG,
          '[SandboxOrchestratorAdapter.processMove] Decision loop iteration:',
          {
            iteration: decisionLoopIteration,
            decisionType: decision.type,
            decisionPlayer: decision.player,
            skipTerritoryAutoResolve: this.skipTerritoryAutoResolve,
            currentPhase: workingState.currentPhase,
          }
        );

        // DEBUG: Trace decision loop for choose_line_option
        debugLog(
          isTestEnvironment() &&
            SANDBOX_ORCHESTRATOR_TRACE_DEBUG &&
            moveToApply.type === 'choose_line_option',
          '[SandboxOrchestratorAdapter] Decision loop:',
          {
            decisionType: decision.type,
            decisionPlayer: decision.player,
            stacksAtDecisionStart: Array.from(workingState.board.stacks.keys()),
          }
        );

        // Chain-capture decisions are handled specially: expose the available
        // continuation moves via getValidMoves() and return without auto-resolving.
        if (decision.type === 'chain_capture') {
          this.chainCaptureOptions = decision.options;
          // eslint-disable-next-line no-console
          console.log('[SandboxOrchestratorAdapter] Setting chainCaptureOptions:', {
            optionsLength: decision.options?.length ?? 0,
            options: decision.options?.map((m) => ({ type: m.type, to: m.to })),
            currentPhase: workingState.currentPhase,
          });
          break;
        }

        // Territory decisions (region_order) are skipped when skipTerritoryAutoResolve
        // is enabled. This is used in replay/traceMode contexts where explicit
        // choose_territory_option moves should come from the recording
        // (legacy alias: process_territory_region).
        if (decision.type === 'region_order' && this.skipTerritoryAutoResolve) {
          break;
        }

        // Line-order decisions are also skipped when skipTerritoryAutoResolve
        // is enabled (trace/replay mode). In that context we want explicit
        // process_line / choose_line_option moves from the recording to drive
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

          const beforeNoAction = workingState;
          result = runProcessTurn(beforeNoAction, noActionMove);
          const afterNoAction = result.nextState;
          const entry = createHistoryEntry(beforeNoAction, afterNoAction, noActionMove, {
            normalizeMoveNumber: true,
          });
          workingState = {
            ...afterNoAction,
            moveHistory: [
              ...beforeNoAction.moveHistory,
              { ...noActionMove, moveNumber: entry.moveNumber },
            ],
            history: [...beforeNoAction.history, entry],
            lastMoveAt: new Date(),
          };
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

        let chosenMove = await delegates.resolveDecision(decision);

        // DEBUG: Trace resolved decision for choose_line_option
        debugLog(
          isTestEnvironment() &&
            SANDBOX_ORCHESTRATOR_TRACE_DEBUG &&
            moveToApply.type === 'choose_line_option',
          '[SandboxOrchestratorAdapter] Decision resolved:',
          {
            chosenMoveType: chosenMove.type,
            chosenMoveFrom: chosenMove.from,
            chosenMoveTo: chosenMove.to,
            eliminationTarget:
              'eliminationTarget' in chosenMove ? chosenMove.eliminationTarget : undefined,
          }
        );

        // WORKAROUND: The shared engine's TerritoryAggregate throws if passed 'forced_elimination',
        // but turnOrchestrator passes it through. We must convert to 'eliminate_rings_from_stack'
        // and rely on ClientSandboxEngine's phase coercion to 'territory_processing' to apply it.
        if (chosenMove.type === 'forced_elimination') {
          chosenMove = {
            ...chosenMove,
            type: 'eliminate_rings_from_stack',
          } as Move;

          // Also coerce the phase in workingState so assertPhaseMoveInvariant accepts it
          if (workingState.currentPhase === 'forced_elimination') {
            workingState = {
              ...workingState,
              currentPhase: 'territory_processing',
            };
          }
        }

        delegates.onProcessingEvent?.({
          type: 'decision_resolved',
          timestamp: new Date(),
          payload: { decision, chosenMove },
        });

        const beforeDecisionMove = workingState;
        result = runProcessTurn(beforeDecisionMove, chosenMove);
        const afterDecisionMove = result.nextState;
        const entry = createHistoryEntry(beforeDecisionMove, afterDecisionMove, chosenMove, {
          normalizeMoveNumber: true,
        });
        workingState = {
          ...afterDecisionMove,
          moveHistory: [
            ...beforeDecisionMove.moveHistory,
            { ...chosenMove, moveNumber: entry.moveNumber },
          ],
          history: [...beforeDecisionMove.history, entry],
          lastMoveAt: new Date(),
        };
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
      const validation = validateMoveWithFSM(state, move);
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

    const validation = validateMoveWithFSM(state, move);
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
   * Uses FSM validation as the canonical validator per RR-CANON-R070.
   */
  public validateMove(move: Move): { valid: boolean; reason?: string } {
    const state = this.stateAccessor.getGameState();
    return validateMoveWithFSM(state, move);
  }

  /**
   * Get all valid moves for the current player and phase.
   *
   * During chain_capture phase, returns the stored continuation options
   * from the previous capture move's pending decision.
   */
  public getValidMoves(): Move[] {
    const state = this.stateAccessor.getGameState();

    // DEBUG: Trace getValidMoves during chain_capture
    if (state.currentPhase === 'chain_capture') {
      // eslint-disable-next-line no-console
      console.log('[SandboxOrchestratorAdapter.getValidMoves] chain_capture phase:', {
        hasChainCaptureOptions: !!this.chainCaptureOptions,
        chainCaptureOptionsLength: this.chainCaptureOptions?.length ?? 0,
        chainCaptureOptionTypes: this.chainCaptureOptions?.map((m) => m.type) ?? [],
        chainCapturePosition: state.chainCapturePosition,
        currentPlayer: state.currentPlayer,
      });
    }

    // During chain_capture phase, return stored continuation options
    if (state.currentPhase === 'chain_capture' && this.chainCaptureOptions) {
      return this.chainCaptureOptions;
    }

    // Delegate to the core orchestrator for interactive moves.
    const interactiveMoves = getValidMoves(state);

    // DEBUG: Trace fallback to core getValidMoves during chain_capture
    if (state.currentPhase === 'chain_capture') {
      // eslint-disable-next-line no-console
      console.log('[SandboxOrchestratorAdapter.getValidMoves] Fallback to core getValidMoves:', {
        interactiveMovesLength: interactiveMoves.length,
        moves: interactiveMoves.map((m) => ({ type: m.type, to: m.to })),
      });
    }

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

      // RR-FIX-2025-12-13: Synthesize no_line_action when in line_processing
      // with no lines. This matches the pending decision surfaced by
      // processPostMovePhases and prevents sandbox AI freeze.
      if (state.currentPhase === 'line_processing') {
        const noLineAction: Move = {
          id: `no-line-action-${moveNumber}`,
          type: 'no_line_action',
          player: state.currentPlayer,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        };
        return [noLineAction];
      }

      // RR-FIX-2025-12-13: Synthesize no_territory_action when in territory_processing
      // with no regions. This matches the pending decision surfaced by
      // processPostMovePhases and prevents sandbox AI freeze.
      if (state.currentPhase === 'territory_processing') {
        const noTerritoryAction: Move = {
          id: `no-territory-action-${moveNumber}`,
          type: 'no_territory_action',
          player: state.currentPlayer,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        };
        return [noTerritoryAction];
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
          const difficultyRaw = playerInfo.aiDifficulty;
          const difficulty =
            typeof difficultyRaw === 'number'
              ? Math.max(1, Math.min(10, Math.round(difficultyRaw)))
              : 5;

          const stateForService = this.stateAccessor.getGameState();
          const serviceResult = await this.tryRequestSandboxAIMove({
            state: serializeGameState(stateForService),
            difficulty,
            playerNumber: decision.player,
          });

          if (serviceResult) {
            const desiredKey = this.moveMatchKey(serviceResult.move);
            const matched = decision.options.find((opt) => this.moveMatchKey(opt) === desiredKey);
            if (matched) {
              recordSandboxAiDiagnostics({
                timestamp: Date.now(),
                gameId: stateForService.id,
                boardType: stateForService.boardType,
                numPlayers: stateForService.players.length,
                playerNumber: decision.player,
                requestedDifficulty: difficulty,
                source: 'service',
                aiType: serviceResult.aiType,
                difficulty: serviceResult.difficulty,
                heuristicProfileId: serviceResult.heuristicProfileId,
                useNeuralNet: serviceResult.useNeuralNet,
                nnModelId: serviceResult.nnModelId,
                nnCheckpoint: serviceResult.nnCheckpoint,
                nnueCheckpoint: serviceResult.nnueCheckpoint,
                thinkingTimeMs: serviceResult.thinkingTimeMs,
              });
              return matched;
            }

            recordSandboxAiDiagnostics({
              timestamp: Date.now(),
              gameId: stateForService.id,
              boardType: stateForService.boardType,
              numPlayers: stateForService.players.length,
              playerNumber: decision.player,
              requestedDifficulty: difficulty,
              source: 'mismatch',
              aiType: serviceResult.aiType,
              difficulty: serviceResult.difficulty,
              heuristicProfileId: serviceResult.heuristicProfileId,
              useNeuralNet: serviceResult.useNeuralNet,
              nnModelId: serviceResult.nnModelId,
              nnCheckpoint: serviceResult.nnCheckpoint,
              nnueCheckpoint: serviceResult.nnueCheckpoint,
              thinkingTimeMs: serviceResult.thinkingTimeMs,
              error: 'service_move_not_in_candidates',
            });
          } else {
            recordSandboxAiDiagnostics({
              timestamp: Date.now(),
              gameId: stateForService.id,
              boardType: stateForService.boardType,
              numPlayers: stateForService.players.length,
              playerNumber: decision.player,
              requestedDifficulty: difficulty,
              source: 'unavailable',
              error: 'service_unavailable',
            });
          }

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

    // RR-FIX-2025-12-27: Create phase-appropriate fallback move based on decision type.
    // Using the wrong move type (e.g., skip_placement in territory_processing) causes
    // phase/move invariant violations that crash or hang the game.
    const state = this.stateAccessor.getGameState();
    const moveNumber = state.moveHistory.length + 1;
    const baseMoveProps = {
      id: `auto-decision-${Date.now()}`,
      player: decision.player,
      to: { x: 0, y: 0 } as Position,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    };

    // Select fallback move type based on decision type and current phase
    switch (decision.type) {
      case 'region_order':
      case 'no_territory_action_required':
        return { ...baseMoveProps, type: 'no_territory_action' };
      case 'line_order':
      case 'no_line_action_required':
        return { ...baseMoveProps, type: 'no_line_action' };
      case 'elimination_target':
        // For ring elimination with no options, we can't eliminate - use no_territory_action
        return { ...baseMoveProps, type: 'no_territory_action' };
      case 'chain_capture':
        // Chain capture with no options means skip capture continuation
        return { ...baseMoveProps, type: 'skip_capture' };
      case 'no_movement_action_required':
        return { ...baseMoveProps, type: 'no_movement_action' };
      case 'no_placement_action_required':
        return { ...baseMoveProps, type: 'no_placement_action' };
      default:
        // Check current phase for context
        if (state.currentPhase === 'territory_processing') {
          return { ...baseMoveProps, type: 'no_territory_action' };
        } else if (state.currentPhase === 'line_processing') {
          return { ...baseMoveProps, type: 'no_line_action' };
        } else if (state.currentPhase === 'movement') {
          return { ...baseMoveProps, type: 'no_movement_action' };
        } else if (state.currentPhase === 'ring_placement') {
          // RR-FIX-2026-01-11: When in ring_placement with 0 rings in hand,
          // use no_placement_action instead of skip_placement
          const currentPlayer = state.players.find((p) => p.id === decision.player);
          if (currentPlayer && currentPlayer.ringsInHand <= 0) {
            return { ...baseMoveProps, type: 'no_placement_action' };
          }
          return { ...baseMoveProps, type: 'skip_placement' };
        }
        // Last resort fallback
        return { ...baseMoveProps, type: 'skip_placement' };
    }
  }

  private moveMatchKey(move: Move): string {
    const positionKey = (pos: unknown): string => {
      if (!pos || typeof pos !== 'object') return '';
      const rec = pos as Record<string, unknown>;
      const x = typeof rec.x === 'number' ? rec.x : '';
      const y = typeof rec.y === 'number' ? rec.y : '';
      const z = typeof rec.z === 'number' ? rec.z : '';
      return `${x},${y},${z}`;
    };

    const positionsKey = (positions: unknown): string => {
      if (!Array.isArray(positions) || positions.length === 0) return '';
      return positions.map((p) => positionKey(p)).join('|');
    };

    const formedLinesKey = (lines: unknown): string => {
      if (!Array.isArray(lines) || lines.length === 0) return '';
      return lines
        .map((line) => {
          if (!line || typeof line !== 'object') return '';
          const l = line as Record<string, unknown>;
          const player = typeof l.player === 'number' ? l.player : '';
          return `${player}:${positionsKey(l.positions)}`;
        })
        .join('||');
    };

    const territoriesKey = (territories: unknown): string => {
      if (!Array.isArray(territories) || territories.length === 0) return '';
      return territories
        .map((territory) => {
          if (!territory || typeof territory !== 'object') return '';
          const t = territory as Record<string, unknown>;
          const controllingPlayer =
            typeof t.controllingPlayer === 'number' ? t.controllingPlayer : '';
          const isDisconnected = t.isDisconnected ? 'd' : 'c';
          return `${controllingPlayer}:${isDisconnected}:${positionsKey(t.spaces)}`;
        })
        .join('||');
    };

    const rec = move as unknown as Record<string, unknown>;
    const buildAmount = typeof rec.buildAmount === 'number' ? rec.buildAmount : '';
    const placementCount = typeof rec.placementCount === 'number' ? rec.placementCount : '';
    const recoveryOption = typeof rec.recoveryOption === 'number' ? rec.recoveryOption : '';

    const eliminationContext =
      typeof rec.eliminationContext === 'string' ? rec.eliminationContext : '';
    const eliminationFromStackPos =
      rec.eliminationFromStack && typeof rec.eliminationFromStack === 'object'
        ? positionKey((rec.eliminationFromStack as Record<string, unknown>).position)
        : '';

    return [
      move.type,
      String(move.player),
      positionKey(move.from),
      positionKey(move.to),
      positionKey(rec['captureTarget']),
      String(buildAmount),
      String(placementCount),
      String(recoveryOption),
      positionsKey(rec['collapsedMarkers']),
      formedLinesKey(rec['formedLines']),
      territoriesKey(rec['disconnectedRegions']),
      eliminationContext,
      eliminationFromStackPos,
    ].join('|');
  }

  private async tryRequestSandboxAIMove(payload: {
    state: ReturnType<typeof serializeGameState>;
    difficulty: number;
    playerNumber: number;
  }): Promise<{
    move: Move;
    evaluation?: unknown;
    thinkingTimeMs?: number | null;
    aiType?: string;
    difficulty?: number;
    heuristicProfileId?: string | null;
    useNeuralNet?: boolean | null;
    nnModelId?: string | null;
    nnCheckpoint?: string | null;
    nnueCheckpoint?: string | null;
  } | null> {
    if (typeof fetch !== 'function') {
      return null;
    }

    // Skip API call in production without AI service configured
    if (!getSandboxAIServiceAvailable()) {
      return null;
    }

    try {
      const response = await fetch('/api/games/sandbox/ai/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        return null;
      }

      const raw = (await response.json()) as unknown;
      if (!raw || typeof raw !== 'object') {
        return null;
      }

      const data = raw as Record<string, unknown>;
      const move = data.move as Move | null | undefined;
      if (!move) {
        return null;
      }

      const aiType = typeof data.aiType === 'string' ? data.aiType : undefined;
      const difficulty = typeof data.difficulty === 'number' ? data.difficulty : undefined;

      const heuristicProfileId =
        typeof data.heuristicProfileId === 'string'
          ? data.heuristicProfileId
          : data.heuristicProfileId === null
            ? null
            : undefined;

      const useNeuralNet =
        typeof data.useNeuralNet === 'boolean'
          ? data.useNeuralNet
          : data.useNeuralNet === null
            ? null
            : undefined;

      const nnModelId =
        typeof data.nnModelId === 'string'
          ? data.nnModelId
          : data.nnModelId === null
            ? null
            : undefined;
      const nnCheckpoint =
        typeof data.nnCheckpoint === 'string'
          ? data.nnCheckpoint
          : data.nnCheckpoint === null
            ? null
            : undefined;
      const nnueCheckpoint =
        typeof data.nnueCheckpoint === 'string'
          ? data.nnueCheckpoint
          : data.nnueCheckpoint === null
            ? null
            : undefined;

      const thinkingTimeMs =
        typeof data.thinkingTimeMs === 'number'
          ? data.thinkingTimeMs
          : data.thinkingTimeMs === null
            ? null
            : undefined;

      return {
        move,
        evaluation: data.evaluation,
        thinkingTimeMs,
        aiType,
        difficulty,
        heuristicProfileId,
        useNeuralNet,
        nnModelId,
        nnCheckpoint,
        nnueCheckpoint,
      };
    } catch {
      return null;
    }
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
