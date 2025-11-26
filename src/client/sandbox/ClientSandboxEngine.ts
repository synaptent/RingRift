import type {
  BoardState,
  BoardType,
  GamePhase,
  GameState,
  GameResult,
  Move,
  Player,
  PlayerType,
  Position,
  RingStack,
  LineInfo,
  RegionOrderChoice,
  GameHistoryEntry,
  Territory,
  CaptureSegmentBoardView,
  MarkerPathHelpers,
  LocalAIRng,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  calculateCapHeight,
  calculateDistance,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
  countRingsInPlayForPlayer,
  canProcessTerritoryRegion,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
  enumerateTerritoryEliminationMoves,
} from '../../shared/engine';
import type { PlayerChoice, PlayerChoiceResponseFor } from '../../shared/types/game';
import { isSandboxAiTraceModeEnabled } from '../../shared/utils/envFlags';
import { SeededRNG, generateGameSeed } from '../../shared/utils/rng';
import { findAllLinesOnBoard } from './sandboxLines';
import { getValidLineProcessingMoves, applyLineDecisionMove } from './sandboxLinesEngine';
import { findDisconnectedRegionsOnBoard } from './sandboxTerritory';
import {
  enumerateSimpleMovementLandings,
  applyMarkerEffectsAlongPathOnBoard,
} from './sandboxMovement';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters,
} from './sandboxCaptures';
import {
  SandboxMovementEngineHooks,
  handleMovementClickSandbox,
  performCaptureChainSandbox,
  enumerateCaptureSegmentsFromSandbox,
} from './sandboxMovementEngine';
import { forceEliminateCapOnBoard } from './sandboxElimination';
import {
  processDisconnectedRegionsForCurrentPlayerEngine,
  TerritoryInteractionHandler,
  getValidTerritoryProcessingMoves,
  applyTerritoryDecisionMove,
} from './sandboxTerritoryEngine';
import {
  SandboxGameEndHooks,
  checkAndApplyVictorySandbox,
  resolveGlobalStalemateIfNeededSandbox,
} from './sandboxGameEnd';
import {
  SandboxTurnState,
  SandboxTurnHooks,
  startTurnForCurrentPlayerSandbox,
  maybeProcessForcedEliminationForCurrentPlayerSandbox,
  advanceTurnAndPhaseForCurrentPlayerSandbox,
} from './sandboxTurnEngine';
import {
  createHypotheticalBoardWithPlacement,
  enumerateLegalRingPlacements,
  hasAnyLegalMoveOrCaptureFrom,
  PlacementBoardView,
} from './sandboxPlacement';
import { maybeRunAITurnSandbox, SandboxAIHooks } from './sandboxAI';
import {
  SandboxOrchestratorAdapter,
  SandboxStateAccessor,
  SandboxDecisionHandler,
  SandboxMoveResult,
} from './SandboxOrchestratorAdapter';

/**
 * Client-local engine harness for the /sandbox route.
 *
 * Scope (current):
 * - Ring placement on non-collapsed, empty cells with no-dead-placement.
 * - Non-capturing movement with distance \u001e stack height and path/marker rules.
 * - Overtaking captures with mandatory chain continuation, using
 *   validateCaptureSegmentOnBoard for legality and a SandboxInteractionHandler
 *   for capture_direction choices when multiple options exist.
 * - Marker behaviour along movement/capture paths aligned with backend TS and Rust.
 * - Line detection & rewards (collapse markers + cap elimination) mirroring
 *   backend behaviour when no interaction manager is wired.
 * - Forced elimination when a player is fully blocked with no rings in hand.
 * - Territory disconnection chain reactions and victory checks (ring
 *   elimination + territory control) per the compact rules.
 */

export type SandboxPlayerKind = PlayerType; // 'human' | 'ai'

export interface SandboxConfig {
  boardType: BoardType;
  numPlayers: number;
  playerKinds: SandboxPlayerKind[]; // indexed 0..3 for players 1..4
}

/**
 * Interaction handler abstraction for sandbox mode. This mirrors the server
 * PlayerInteractionHandler + PlayerInteractionManager model, but runs entirely
 * on the client. For now we only need PlayerChoice support; as the sandbox
 * grows, this can be expanded.
 */
export interface SandboxInteractionHandler {
  requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>>;
}

export interface ClientSandboxEngineOptions {
  config: SandboxConfig;
  interactionHandler: SandboxInteractionHandler;
  /** Optional test-only flag: when true, the engine may enable additional
   *  trace/replay behaviours for parity harnesses without affecting normal
   *  sandbox UX.
   */
  traceMode?: boolean;
}

export class ClientSandboxEngine {
  private gameState: GameState;
  private interactionHandler: SandboxInteractionHandler;
  // When true, the engine is running under a trace/replay harness. This
  // is currently reserved for future parity-specific behaviour and does
  // not alter normal sandbox rules or AI policy.
  private readonly traceMode: boolean;

  // ═══════════════════════════════════════════════════════════════════════
  // Orchestrator Adapter Integration
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Feature flag: when true, applyCanonicalMoveInternal delegates to the
   * shared orchestrator via SandboxOrchestratorAdapter instead of using
   * the legacy sandbox-specific logic.
   *
   * Default: false (parity issues discovered in Phase 6 - see LEGACY_CODE_ELIMINATION_PLAN.md)
   *
   * Known parity issues blocking full elimination:
   * - "Landing on own marker eliminates bottom ring" rule not triggered via adapter
   * - Some movement tests require legacy path for correct semantics
   *
   * Enable explicitly for testing adapter behavior:
   *   engine.enableOrchestratorAdapter();
   */
  private useOrchestratorAdapter: boolean = false;

  /** Lazily-initialized adapter instance */
  private orchestratorAdapter: SandboxOrchestratorAdapter | null = null;

  // Per-game RNG for deterministic AI behavior
  private rng: SeededRNG;

  // When non-null, the sandbox game has ended with this result.
  private victoryResult: GameResult | null = null;

  // Internal turn-level state for sandbox per-turn flow.
  private _hasPlacedThisTurn: boolean = false;
  private _mustMoveFromStackKey: string | undefined;

  // Internal selection state for movement. This is intentionally kept off of
  // GameState to avoid diverging the shared type.
  private _selectedStackKey: string | undefined;

  // Internal flag used to distinguish between human-initiated movement
  // (click-driven) and canonical replay via applyCanonicalMove. This lets
  // us reuse the same movement engine while avoiding double history
  // entries for canonical moves.
  private _movementInvocationContext: 'human' | 'canonical' | null = null;

  // Test-only: last logical AI move chosen by maybeRunAITurn. This is used
  // by backend-vs-sandbox debug harnesses to map sandbox actions into a
  // canonical Move shape for comparison against backend getValidMoves.
  private _lastAIMove: Move | null = null;

  // Internal flag used in move-driven line decision phases to indicate that
  // the current player has collapsed a line in a way that requires a
  // mandatory ring elimination via an explicit eliminate_rings_from_stack
  // Move. This mirrors the backend GameEngine.pendingLineRewardElimination
  // flag but remains local to the sandbox engine.
  private _pendingLineRewardElimination: boolean = false;

  // Internal flag used in move-driven territory decision phases to indicate
  // that the current player has processed at least one disconnected region
  // and therefore owes a mandatory self-elimination via an explicit
  // eliminate_rings_from_stack decision. This mirrors the backend
  // GameEngine.pendingTerritorySelfElimination flag but remains local to
  // the sandbox engine.
  private _pendingTerritorySelfElimination: boolean = false;

  // Test-only checkpoint hook used by parity/diagnostic harnesses to capture
  // GameState snapshots at key points inside canonical move application and
  // post-movement processing. When unset, all debugCheckpoint calls are no-ops.
  /**
   * Host-internal metadata for last-player-standing (R172) detection.
   * These fields track, per round, which players had any real actions
   * available (placement, non-capture movement, or overtaking capture)
   * at the start of their most recent interactive turn. They are kept
   * off of GameState so sandbox snapshots and wire formats remain
   * unchanged.
   */
  private _lpsRoundIndex: number = 0;
  private _lpsCurrentRoundActorMask: Map<number, boolean> = new Map();
  private _lpsCurrentRoundFirstPlayer: number | null = null;
  private _lpsExclusivePlayerForCompletedRound: number | null = null;

  private _debugCheckpointHook?: ((label: string, state: GameState) => void) | undefined;

  /**
   * Internal helper to record a single capture segment as a canonical
   * Move + GameHistoryEntry, mirroring backend GameEngine semantics.
   * The initial segment in a chain is represented as an
   * 'overtaking_capture' move; all follow-up segments use
   * 'continue_capture_segment'. For the final segment in a chain, the
   * "after" snapshot observed by history includes post-movement
   * automatic consequences (lines, territory, victory, next-player),
   * just like the backend's structured history.
   */
  private async handleCaptureSegmentApplied(info: {
    before: GameState;
    after: GameState;
    from: Position;
    target: Position;
    landing: Position;
    playerNumber: number;
    segmentIndex: number;
    isFinal: boolean;
  }): Promise<void> {
    const moveType: Move['type'] =
      info.segmentIndex === 0 ? 'overtaking_capture' : 'continue_capture_segment';

    const moveNumber = this.gameState.history.length + 1;

    const move: Move = {
      id: '',
      type: moveType,
      player: info.playerNumber,
      from: info.from,
      to: info.landing,
      captureTarget: info.target,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move;

    // For intermediate segments, the GameState snapshot observed by
    // appendHistoryEntry will reflect board state immediately after the
    // segment is applied. For the final segment, performCaptureChainSandbox
    // ensures that onMovementComplete has already run before this callback
    // is invoked, so getGameState() includes post-movement consequences
    // (lines, territory, victory, next-player), matching backend history.
    this.appendHistoryEntry(info.before, move);
  }

  /**
   * Internal helper to record a simple non-capturing movement as a canonical
   * move + history entry, mirroring backend GameEngine semantics for
   * move_stack. This is currently used for human-driven movement clicks;
   * canonical replays continue to record history via applyCanonicalMove.
   */
  private async handleSimpleMoveApplied(info: {
    before: GameState;
    after: GameState;
    from: Position;
    landing: Position;
    playerNumber: number;
  }): Promise<void> {
    const moveNumber = this.gameState.history.length + 1;

    const move: Move = {
      id: '',
      type: 'move_stack',
      player: info.playerNumber,
      from: info.from,
      to: info.landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move;

    this.appendHistoryEntry(info.before, move);
  }

  /**
   * Append a structured history entry for a canonical move applied to the
   * sandbox game state. This mirrors the backend GameEngine
   * appendHistoryEntry but runs entirely client-side and is primarily used
   * by parity/debug tooling.
   */
  private appendHistoryEntry(before: GameState, action: Move): void {
    const after = this.getGameState();
    const progressBefore = computeProgressSnapshot(before);
    const progressAfter = computeProgressSnapshot(after);

    // Normalise moveNumber so that sandbox history always uses a contiguous
    // 1..N sequence regardless of how callers populated Move.moveNumber.
    const nextMoveNumber = this.gameState.history.length + 1;

    const normalizedAction: Move = {
      ...action,
      moveNumber: nextMoveNumber,
    };

    const entry: GameHistoryEntry = {
      moveNumber: nextMoveNumber,
      action: normalizedAction,
      actor: normalizedAction.player,
      phaseBefore: before.currentPhase,
      phaseAfter: after.currentPhase,
      statusBefore: before.gameStatus,
      statusAfter: after.gameStatus,
      progressBefore,
      progressAfter,
      stateHashBefore: hashGameState(before),
      stateHashAfter: hashGameState(after),
      boardBeforeSummary: summarizeBoard(before.board),
      boardAfterSummary: summarizeBoard(after.board),
    };

    this.gameState = {
      ...this.gameState,
      history: [...this.gameState.history, entry],
    };
  }

  constructor(opts: ClientSandboxEngineOptions) {
    const { config, interactionHandler, traceMode } = opts;
    this.interactionHandler = interactionHandler;
    this.traceMode = traceMode ?? isSandboxAiTraceModeEnabled();

    // Initialize with temporary seed; will be set from gameState if provided
    this.rng = new SeededRNG(generateGameSeed());

    const board = this.createEmptyBoard(config.boardType);
    const players: Player[] = Array.from({ length: config.numPlayers }, (_, idx) => {
      const playerNumber = idx + 1;
      const kind = config.playerKinds[idx] ?? 'human';
      return {
        id: `sandbox-${playerNumber}`,
        username: `Player ${playerNumber}`,
        type: kind,
        playerNumber,
        isReady: true,
        timeRemaining: 0,
        aiDifficulty: kind === 'ai' ? 5 : undefined,
        ringsInHand: BOARD_CONFIGS[config.boardType].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      };
    });

    const now = new Date();
    const boardConfig = BOARD_CONFIGS[config.boardType];

    // Generate seed for this sandbox game
    const gameSeed = generateGameSeed();
    this.rng = new SeededRNG(gameSeed);

    this.gameState = {
      id: 'sandbox-local',
      boardType: config.boardType,
      rngSeed: gameSeed,
      board,
      players,
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      },
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: false,
      maxPlayers: config.numPlayers,
      totalRingsInPlay: boardConfig.ringsPerPlayer * config.numPlayers,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor((boardConfig.ringsPerPlayer * config.numPlayers) / 2) + 1,
      territoryVictoryThreshold: Math.floor(boardConfig.totalSpaces / 2) + 1,
    };

    // Initialize orchestrator adapter lazily when first needed
    this.orchestratorAdapter = null;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Orchestrator Adapter Public API
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Enable delegation to the shared orchestrator adapter.
   *
   * When enabled, applyCanonicalMoveInternal() will delegate rules logic
   * to the shared orchestrator (processTurnAsync), keeping only sandbox-
   * specific concerns (state management, decision UI, callbacks) local.
   *
   * This is designed for:
   * - Gradual rollout and validation of the unified orchestrator
   * - Parity testing between legacy sandbox logic and orchestrator
   * - Future removal of duplicated sandbox rules code
   *
   * Usage:
   * ```typescript
   * const engine = new ClientSandboxEngine(opts);
   * engine.enableOrchestratorAdapter();
   * await engine.applyCanonicalMove(move); // Now uses orchestrator
   * ```
   */
  public enableOrchestratorAdapter(): void {
    this.useOrchestratorAdapter = true;
    // Ensure adapter is initialized
    this.getOrchestratorAdapter();
  }

  /**
   * Disable delegation to the shared orchestrator adapter.
   * Returns to using legacy sandbox-specific rules logic.
   */
  public disableOrchestratorAdapter(): void {
    this.useOrchestratorAdapter = false;
  }

  /**
   * Check whether the orchestrator adapter is enabled.
   */
  public isOrchestratorAdapterEnabled(): boolean {
    return this.useOrchestratorAdapter;
  }

  /**
   * Get or create the orchestrator adapter instance.
   */
  private getOrchestratorAdapter(): SandboxOrchestratorAdapter {
    if (!this.orchestratorAdapter) {
      this.orchestratorAdapter = this.createOrchestratorAdapter();
    }
    return this.orchestratorAdapter;
  }

  /**
   * Create a new SandboxOrchestratorAdapter wired to this engine.
   */
  private createOrchestratorAdapter(): SandboxOrchestratorAdapter {
    const stateAccessor: SandboxStateAccessor = {
      getGameState: () => this.getGameState(),
      updateGameState: (state: GameState) => {
        this.gameState = state;
      },
      getPlayerInfo: (playerId: string): { type: 'human' | 'ai'; aiDifficulty?: number } => {
        // Extract player number from playerId (format: "player-N" or "sandbox-N")
        const match = playerId.match(/(\d+)$/);
        const playerNumber = match ? parseInt(match[1], 10) : 1;
        const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
        if (player?.type === 'ai') {
          return { type: 'ai', aiDifficulty: player.aiDifficulty };
        }
        return { type: 'human' };
      },
    };

    const decisionHandler: SandboxDecisionHandler = {
      requestDecision: async (decision) => {
        // Map PendingDecision to sandbox interaction handler format
        // The sandbox interaction handler uses PlayerChoice format
        const playerChoice = this.mapPendingDecisionToPlayerChoice(decision);
        const response = await this.interactionHandler.requestChoice(playerChoice);
        return this.mapPlayerChoiceResponseToMove(decision, response);
      },
    };

    return new SandboxOrchestratorAdapter({
      stateAccessor,
      decisionHandler,
      callbacks: {
        debugHook: this._debugCheckpointHook
          ? (label: string, state: GameState) => {
              this._debugCheckpointHook!(label, state);
            }
          : undefined,
        onError: (error: Error, context: string) => {
          // eslint-disable-next-line no-console
          console.error(`[SandboxOrchestratorAdapter] Error in ${context}:`, error);
        },
      },
    });
  }

  /**
   * Map a PendingDecision from the orchestrator to a PlayerChoice for
   * the sandbox interaction handler.
   */
  private mapPendingDecisionToPlayerChoice(decision: any): PlayerChoice {
    const decisionType = decision.type;
    const options = decision.options as Move[];

    switch (decisionType) {
      case 'line_order':
        return {
          id: `sandbox-line-order-${Date.now()}`,
          gameId: this.gameState.id,
          playerNumber: decision.player,
          type: 'line_order',
          prompt: 'Select which line to process first',
          options: options.map((opt: Move, idx: number) => ({
            lineId: `line-${idx}`,
            markerPositions: opt.formedLines?.[0]?.positions ?? [],
            moveId: opt.id || `move-${idx}`,
          })),
        };

      case 'line_reward':
        return {
          id: `sandbox-line-reward-${Date.now()}`,
          gameId: this.gameState.id,
          playerNumber: decision.player,
          type: 'line_reward_option',
          prompt: 'Choose reward for overlength line',
          options: options.map((opt: Move) => {
            const isCollapseAll =
              opt.collapsedMarkers?.length === opt.formedLines?.[0]?.positions?.length;
            return isCollapseAll
              ? ('option_1_collapse_all_and_eliminate' as const)
              : ('option_2_min_collapse_no_elimination' as const);
          }),
          moveIds: options.reduce(
            (acc, opt: Move, idx: number) => {
              const isCollapseAll =
                opt.collapsedMarkers?.length === opt.formedLines?.[0]?.positions?.length;
              const key = isCollapseAll
                ? 'option_1_collapse_all_and_eliminate'
                : 'option_2_min_collapse_no_elimination';
              acc[key] = opt.id || `move-${idx}`;
              return acc;
            },
            {} as Record<string, string>
          ),
        };

      case 'region_order':
        return {
          id: `sandbox-region-order-${Date.now()}`,
          gameId: this.gameState.id,
          playerNumber: decision.player,
          type: 'region_order',
          prompt: 'Select which region to process first',
          options: options.map((opt: Move, idx: number) => {
            const region = opt.disconnectedRegions?.[0];
            return {
              regionId: `region-${idx}`,
              size: region?.spaces?.length ?? 0,
              representativePosition: region?.spaces?.[0] ?? { x: 0, y: 0 },
              moveId: opt.id || `move-${idx}`,
            };
          }),
        };

      case 'capture_direction':
        return {
          id: `sandbox-capture-${Date.now()}`,
          gameId: this.gameState.id,
          playerNumber: decision.player,
          type: 'capture_direction',
          prompt: 'Select capture direction',
          options: options.map((opt: Move) => ({
            targetPosition: opt.captureTarget!,
            landingPosition: opt.to,
            capturedCapHeight:
              this.gameState.board.stacks.get(positionToString(opt.captureTarget!))?.capHeight ?? 0,
          })),
        };

      default:
        // Generic fallback - use line_order as default
        return {
          id: `sandbox-decision-${Date.now()}`,
          gameId: this.gameState.id,
          playerNumber: decision.player,
          type: 'line_order',
          prompt: String(decision.type),
          options: [],
        };
    }
  }

  /**
   * Map a PlayerChoice response back to a Move for the orchestrator.
   */
  private mapPlayerChoiceResponseToMove(decision: any, response: any): Move {
    // Find the matching option from the original decision
    const options = decision.options as Move[];

    // Try to match based on response content
    if (response.selectedLineIndex !== undefined && options[response.selectedLineIndex]) {
      return options[response.selectedLineIndex];
    }

    if (response.selectedRegionIndex !== undefined && options[response.selectedRegionIndex]) {
      return options[response.selectedRegionIndex];
    }

    if (response.selectedOption) {
      // Match by position for capture direction
      const selected = options.find((opt: Move) => {
        if (opt.captureTarget && response.selectedOption.targetPosition) {
          return (
            positionToString(opt.captureTarget) ===
              positionToString(response.selectedOption.targetPosition) &&
            positionToString(opt.to!) === positionToString(response.selectedOption.landingPosition)
          );
        }
        return false;
      });
      if (selected) return selected;
    }

    // Default to first option
    return options[0];
  }

  /**
   * Process a move via the orchestrator adapter.
   *
   * This method delegates to the shared orchestrator for rules logic,
   * handling sandbox-specific concerns like history recording.
   */
  private async processMoveViaAdapter(
    move: Move,
    beforeStateForHistory: GameState
  ): Promise<boolean> {
    const adapter = this.getOrchestratorAdapter();
    const result = await adapter.processMove(move);

    if (!result.success) {
      return false;
    }

    // Update victory result if game ended
    if (result.victoryResult) {
      this.victoryResult = result.victoryResult;
    }

    // Update internal turn state based on move type
    if (move.type === 'place_ring' && move.to) {
      this._hasPlacedThisTurn = true;
      this._mustMoveFromStackKey = positionToString(move.to);
      this._selectedStackKey = positionToString(move.to);
    } else if (
      move.type === 'move_stack' ||
      move.type === 'move_ring' ||
      move.type === 'overtaking_capture' ||
      move.type === 'continue_capture_segment'
    ) {
      this._selectedStackKey = undefined;
    }

    // Clear pending flags based on move type
    if (move.type === 'eliminate_rings_from_stack') {
      this._pendingLineRewardElimination = false;
      this._pendingTerritorySelfElimination = false;
    }

    return result.metadata?.stateChanged ?? true;
  }

  /**
   * Test-only helper: register a debug checkpoint hook so parity/diagnostic
   * harnesses can capture GameState snapshots at key points inside canonical
   * move application and post-movement processing.
   */
  public setDebugCheckpointHook(
    hook: ((label: string, state: GameState) => void) | undefined
  ): void {
    this._debugCheckpointHook = hook;
  }

  private debugCheckpoint(label: string): void {
    if (this._debugCheckpointHook) {
      this._debugCheckpointHook(label, this.getGameState());
    }
  }

  /**
   * Return a defensive snapshot of the current GameState.
   *
   * Unlike the earliest sandbox version, this now deep-clones the board's
   * Map/array fields so that parity/debug tooling (and any callers holding
   * onto past snapshots) see stable pre/post views rather than aliases that
   * are mutated by subsequent moves. This mirrors the backend
   * GameEngine.getGameState semantics.
   */
  public getGameState(): GameState {
    const state = this.gameState;
    const board = state.board;

    const clonedBoard: BoardState = {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };

    return {
      ...state,
      board: clonedBoard,
      moveHistory: [...state.moveHistory],
      history: [...state.history],
      players: state.players.map((p) => ({ ...p })),
      spectators: [...state.spectators],
    };
  }

  /**
   * When non-null, contains the terminal GameResult for this sandbox game.
   * This mirrors the backend GameContext.victoryState shape so the
   * VictoryModal component can be reused for local games.
   */
  public getVictoryResult(): GameResult | null {
    return this.victoryResult;
  }

  /**
   * Test-only helper: expose the last logical AI move chosen by
   * maybeRunAITurn in a canonical Move shape. This is used by
   * backend-vs-sandbox debug harnesses to validate sandbox AI
   * decisions against backend getValidMoves.
   */
  public getLastAIMoveForTesting(): Move | null {
    return this._lastAIMove ? { ...this._lastAIMove } : null;
  }

  /**
   * Clear any internal movement selection state. This is used by the sandbox
   * UI when it wants to discard a previous selection and treat the next click
   * as a fresh source-selection, keeping BoardView highlights and engine
   * semantics aligned.
   */
  public clearSelection(): void {
    this._selectedStackKey = undefined;
  }

  /**
   * Handle a human click on a board cell in sandbox mode. This is the main
   * entry point for the /sandbox UI, analogous to the backend click-to-move
   * flow in GamePage, but targeting the local GameState instead of the
   * WebSocket server.
   *
   * Ring placement is now routed through the same canonical Move-applier used
   * by parity harnesses so that human and AI actions share a single semantic
   * path. Movement clicks continue to delegate to handleMovementClick.
   */
  public async handleHumanCellClick(pos: Position): Promise<void> {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    if (this.gameState.currentPhase === 'ring_placement') {
      const beforeState = this.getGameState();
      const playerNumber = beforeState.currentPlayer;

      // Preserve placed-on-stack metadata for history, mirroring the backend
      // place_ring representation.
      const key = positionToString(pos);
      const existingBefore = beforeState.board.stacks.get(key);
      const placedOnStack = !!existingBefore && existingBefore.rings.length > 0;

      const moveNumber = beforeState.history.length + 1;

      const move: Move = {
        id: '',
        type: 'place_ring',
        player: playerNumber,
        to: pos,
        placementCount: 1,
        placedOnStack,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;

      // Apply via the canonical Move-applier with no-dead-placement enforced,
      // so that human sandbox placements share the same semantics as AI
      // placements and backend RuleEngine validation.
      const changed = await this.applyCanonicalMoveInternal(move, {
        bypassNoDeadPlacement: false,
      });

      if (!changed) {
        return;
      }

      this.appendHistoryEntry(beforeState, move);
    } else if (this.gameState.currentPhase === 'movement') {
      // Human-driven movement click. Record canonical history via the
      // movement engine hooks without interfering with canonical replays.
      this._movementInvocationContext = 'human';
      try {
        await this.handleMovementClick(pos);
      } finally {
        this._movementInvocationContext = null;
      }
    }
  }

  /**
   * Run a single AI turn in sandbox mode.
   *
   * Behaviour:
   * - In ring_placement: chooses a random legal placement that satisfies
   *   no-dead-placement (if it still has rings in hand).
   * - In movement:
   *   - Prefer an overtaking capture chain when at least one capture
   *     segment exists for any of the current player's stacks.
   *   - Otherwise choose a random simple non-capturing move.
   *
   * This keeps local games progressing while remaining aligned with the
   * backend RuleEngine semantics (movement reachability + capture chains).
   */
  public async maybeRunAITurn(rng?: LocalAIRng): Promise<void> {
    // Use provided RNG if given (for testing), otherwise use instance RNG
    const effectiveRng = rng ?? (() => this.rng.next());

    const hooks: SandboxAIHooks = {
      getPlayerStacks: (playerNumber: number, board: BoardState) =>
        this.getPlayerStacks(playerNumber, board),
      hasAnyLegalMoveOrCaptureFrom: (from: Position, playerNumber: number, board: BoardState) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      enumerateLegalRingPlacements: (playerNumber: number) =>
        this.enumerateLegalRingPlacements(playerNumber),
      createHypotheticalBoardWithPlacement: (
        board: BoardState,
        position: Position,
        playerNumber: number,
        count: number = 1
      ): BoardState =>
        this.createHypotheticalBoardWithPlacement(board, position, playerNumber, count),
      tryPlaceRings: async (position: Position, count: number) =>
        await this.tryPlaceRings(position, count),
      enumerateCaptureSegmentsFrom: (from: Position, playerNumber: number) =>
        this.enumerateCaptureSegmentsFrom(from, playerNumber),
      enumerateSimpleMovementLandings: (playerNumber: number) =>
        this.enumerateSimpleMovementLandings(playerNumber),
      maybeProcessForcedEliminationForCurrentPlayer: () =>
        this.maybeProcessForcedEliminationForCurrentPlayer(),
      handleMovementClick: (position: Position) => this.handleMovementClick(position),
      appendHistoryEntry: (before: GameState, action: Move) =>
        this.appendHistoryEntry(before, action),
      getGameState: () => this.getGameState(),
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      setLastAIMove: (move: Move | null) => {
        this._lastAIMove = move;
      },
      setSelectedStackKey: (key: string | undefined) => {
        this._selectedStackKey = key;
      },
      getMustMoveFromStackKey: () => this._mustMoveFromStackKey,
      applyCanonicalMove: (move: Move) => this.applyCanonicalMove(move),
      hasPendingTerritorySelfElimination: () => this._pendingTerritorySelfElimination,
      hasPendingLineRewardElimination: () => this._pendingLineRewardElimination,
    };

    await maybeRunAITurnSandbox(hooks, effectiveRng);
  }
  /**
   * Get all valid landing positions for the current player from the given
   * source position. This is used by the UI to highlight valid targets.
   */
  public getValidLandingPositionsForCurrentPlayer(from: Position): Position[] {
    const playerNumber = this.gameState.currentPlayer;
    const fromKey = positionToString(from);

    // 1. Check for capture segments
    const captureSegments = this.enumerateCaptureSegmentsFrom(from, playerNumber);
    if (captureSegments.length > 0) {
      return captureSegments.map((seg) => seg.landing);
    }

    // 2. Check for simple movement
    const simpleMoves = this.enumerateSimpleMovementLandings(playerNumber);
    return simpleMoves.filter((m) => m.fromKey === fromKey).map((m) => m.to);
  }

  /**
   * Enumerate legal ring placement positions for the given player, enforcing
   * the same no-dead-placement rule used for human placement clicks.
   *
   * Unlike the earliest sandbox version, placement is now allowed on both
   * empty spaces and existing stacks (non-collapsed only). Stacking uses the
   * same semantics as createHypotheticalBoardWithPlacement.
   */
  private enumerateLegalRingPlacements(playerNumber: number): Position[] {
    const view: PlacementBoardView = {
      isValidPosition: (pos) => this.isValidPosition(pos),
      isCollapsedSpace: (pos, board) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos, board) => this.getMarkerOwner(pos, board),
    };

    const state = this.gameState;
    const player = state.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.ringsInHand <= 0) {
      return [];
    }

    const boardType = state.boardType;
    const boardConfig = BOARD_CONFIGS[boardType];

    // Own-colour supply accounting: total rings of this player's colour
    // currently in play (on the board in any stack, regardless of control,
    // plus in hand) must never exceed ringsPerPlayer. We derive an
    // own-colour board count for the placement context from this.
    const totalInPlay = countRingsInPlayForPlayer(state, playerNumber);
    const ringsOnBoard = totalInPlay - player.ringsInHand;
    const remainingByCap = boardConfig.ringsPerPlayer - ringsOnBoard;
    const remainingBySupply = player.ringsInHand;
    const maxAvailableGlobal = Math.min(remainingByCap, remainingBySupply);

    if (maxAvailableGlobal <= 0) {
      return [];
    }

    const ctx = {
      boardType,
      player: playerNumber,
      ringsInHand: player.ringsInHand,
      ringsPerPlayerCap: boardConfig.ringsPerPlayer,
      ringsOnBoard,
      maxAvailableGlobal,
    };

    return enumerateLegalRingPlacements(boardType, state.board, playerNumber, view, ctx);
  }

  /**
   * Enumerate simple, non-capturing movement options for the given player.
   * This mirrors the path/occupancy checks in handleMovementClick but treats
   * all legal landing positions as candidates for AI selection.
   */
  private enumerateSimpleMovementLandings(playerNumber: number): {
    fromKey: string;
    to: Position;
  }[] {
    return enumerateSimpleMovementLandings(
      this.gameState.boardType,
      this.gameState.board,
      playerNumber,
      (pos: Position) => this.isValidPosition(pos)
    );
  }

  /**
   * R172 helper: true if the given player has any real action available
   * at the start of their turn (placement, non-capture movement, or
   * overtaking capture). Forced elimination and decision moves do not
   * count as real actions.
   */
  private hasAnyRealActionForPlayer(playerNumber: number): boolean {
    if (this.gameState.gameStatus !== 'active') {
      return false;
    }

    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player) {
      return false;
    }

    if (!this.playerHasMaterial(playerNumber)) {
      return false;
    }

    // Ring placement
    const placements = this.enumerateLegalRingPlacements(playerNumber);
    if (placements.length > 0) {
      return true;
    }

    // Non-capture movement
    const simpleMoves = this.enumerateSimpleMovementLandings(playerNumber);
    if (simpleMoves.length > 0) {
      return true;
    }

    // Overtaking capture from any controlled stack
    const board = this.gameState.board;
    for (const stack of board.stacks.values()) {
      if (stack.controllingPlayer !== playerNumber) continue;
      const segments = this.enumerateCaptureSegmentsFrom(stack.position, playerNumber);
      if (segments.length > 0) {
        return true;
      }
    }

    return false;
  }

  /**
   * True if the player has any material left: at least one controlled
   * stack or at least one ring in hand.
   */
  private playerHasMaterial(playerNumber: number): boolean {
    const board = this.gameState.board;
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    const hasStacks = Array.from(board.stacks.values()).some(
      (stack) => stack.controllingPlayer === playerNumber
    );
    const ringsInHand = player?.ringsInHand ?? 0;
    return hasStacks || ringsInHand > 0;
  }

  /**
   * Update last-player-standing round tracking for the current player.
   *
   * This mirrors the backend and Python LPS helpers by:
   * - Computing whether the current player has any real actions.
   * - Recording that fact in _lpsCurrentRoundActorMask.
   * - When all active players have been seen in this round, determining
   *   whether exactly one player had real actions and, if so, recording
   *   them as _lpsExclusivePlayerForCompletedRound.
   */
  private updateLpsRoundTrackingForCurrentPlayer(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const state = this.gameState;
    const current = state.currentPlayer;

    const activePlayers = state.players
      .filter((p) => this.playerHasMaterial(p.playerNumber))
      .map((p) => p.playerNumber);

    if (activePlayers.length === 0) {
      return;
    }

    const activeSet = new Set(activePlayers);
    if (!activeSet.has(current)) {
      return;
    }

    const first = this._lpsCurrentRoundFirstPlayer;
    const startingNewCycle = first === null || !activeSet.has(first);

    if (startingNewCycle) {
      this._lpsRoundIndex += 1;
      this._lpsCurrentRoundFirstPlayer = current;
      this._lpsCurrentRoundActorMask.clear();
      this._lpsExclusivePlayerForCompletedRound = null;
    } else if (current === first && this._lpsCurrentRoundActorMask.size > 0) {
      // Completed the previous round; finalise it before starting a new one.
      this.finalizeCompletedLpsRound(activePlayers);
      this._lpsRoundIndex += 1;
      this._lpsCurrentRoundActorMask.clear();
      this._lpsCurrentRoundFirstPlayer = current;
    }

    const hasRealAction = this.hasAnyRealActionForPlayer(current);
    this._lpsCurrentRoundActorMask.set(current, hasRealAction);
  }

  /**
   * Finalise a completed round by determining whether exactly one active
   * player had any real actions throughout the round.
   */
  private finalizeCompletedLpsRound(activePlayers: number[]): void {
    const truePlayers: number[] = [];
    for (const pid of activePlayers) {
      if (this._lpsCurrentRoundActorMask.get(pid)) {
        truePlayers.push(pid);
      }
    }

    if (truePlayers.length === 1) {
      this._lpsExclusivePlayerForCompletedRound = truePlayers[0];
    } else {
      this._lpsExclusivePlayerForCompletedRound = null;
    }
  }

  /**
   * Build a GameResult for a last-player-standing victory, mirroring the
   * sandboxVictory score aggregation used for other victory reasons.
   */
  private buildLastPlayerStandingResult(winner: number): GameResult {
    const state = this.gameState;
    const board = state.board;

    const perPlayer: {
      [playerNumber: number]: {
        ringsRemaining: number;
        territorySpaces: number;
        ringsEliminated: number;
      };
    } = {};

    for (const p of state.players) {
      perPlayer[p.playerNumber] = {
        ringsRemaining: 0,
        territorySpaces: p.territorySpaces,
        ringsEliminated: p.eliminatedRings,
      };
    }

    for (const stack of board.stacks.values()) {
      const owner = stack.controllingPlayer;
      const entry = perPlayer[owner];
      if (entry) {
        entry.ringsRemaining += stack.stackHeight;
      }
    }

    const ringsRemaining: { [playerNumber: number]: number } = {};
    const territorySpaces: { [playerNumber: number]: number } = {};
    const ringsEliminated: { [playerNumber: number]: number } = {};

    for (const p of state.players) {
      const entry = perPlayer[p.playerNumber];
      ringsRemaining[p.playerNumber] = entry ? entry.ringsRemaining : 0;
      territorySpaces[p.playerNumber] = entry ? entry.territorySpaces : 0;
      ringsEliminated[p.playerNumber] = entry ? entry.ringsEliminated : 0;
    }

    return {
      winner,
      reason: 'last_player_standing',
      finalScore: {
        ringsEliminated,
        territorySpaces,
        ringsRemaining,
      },
    };
  }

  /**
   * Check whether R172 is satisfied at the start of the current player's
   * interactive turn and, if so, end the sandbox game with an LPS result.
   */
  private maybeEndGameByLastPlayerStanding(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const phase = this.gameState.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return;
    }

    const candidate = this._lpsExclusivePlayerForCompletedRound;
    if (candidate == null) {
      return;
    }

    const state = this.gameState;
    if (state.currentPlayer !== candidate) {
      return;
    }

    if (!this.hasAnyRealActionForPlayer(candidate)) {
      // Candidate no longer has real actions; require a fresh qualifying round.
      this._lpsExclusivePlayerForCompletedRound = null;
      return;
    }

    const board = state.board;
    let othersHaveActions = false;

    for (const p of state.players) {
      if (p.playerNumber === candidate) {
        continue;
      }
      if (!this.playerHasMaterial(p.playerNumber)) {
        continue;
      }
      if (this.hasAnyRealActionForPlayer(p.playerNumber)) {
        othersHaveActions = true;
        break;
      }
    }

    if (othersHaveActions) {
      // Another player with material regained a real action before R172 fired.
      this._lpsExclusivePlayerForCompletedRound = null;
      return;
    }

    const result = this.buildLastPlayerStandingResult(candidate);

    this.gameState = {
      ...state,
      gameStatus: 'completed',
      winner: candidate,
      // Normalise terminal phase away from decision phases for UI/parity.
      currentPhase: 'ring_placement',
    };
    this.victoryResult = result;

    // Reset round tracking after a terminal LPS outcome.
    this._lpsCurrentRoundActorMask.clear();
    this._lpsCurrentRoundFirstPlayer = null;
    this._lpsExclusivePlayerForCompletedRound = null;
  }

  /**
   * Host-level hook invoked whenever a new interactive turn begins for
   * the current player. This is where LPS tracking and checks are wired
   * into the sandbox turn lifecycle.
   */
  private handleStartOfInteractiveTurn(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const phase = this.gameState.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return;
    }

    const beforeSnapshot = {
      currentPlayer: this.gameState.currentPlayer,
      currentPhase: this.gameState.currentPhase,
      gameStatus: this.gameState.gameStatus,
      lpsRoundIndex: this._lpsRoundIndex,
      lpsCurrentRoundFirstPlayer: this._lpsCurrentRoundFirstPlayer,
      lpsExclusivePlayerForCompletedRound: this._lpsExclusivePlayerForCompletedRound,
    };

    this.updateLpsRoundTrackingForCurrentPlayer();
    this.maybeEndGameByLastPlayerStanding();

    const afterSnapshot = {
      currentPlayer: this.gameState.currentPlayer,
      currentPhase: this.gameState.currentPhase,
      gameStatus: this.gameState.gameStatus,
      lpsRoundIndex: this._lpsRoundIndex,
      lpsCurrentRoundFirstPlayer: this._lpsCurrentRoundFirstPlayer,
      lpsExclusivePlayerForCompletedRound: this._lpsExclusivePlayerForCompletedRound,
    };

    if (
      typeof process !== 'undefined' &&
      (process as any).env &&
      (process as any).env.NODE_ENV === 'test'
    ) {
      // eslint-disable-next-line no-console
      console.log('[TurnTrace.sandbox.handleStartOfInteractiveTurn]', {
        decision: 'handleStartOfInteractiveTurn',
        reason: 'start_interactive_turn',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
    }
  }

  // === Internal helpers ===

  private createEmptyBoard(boardType: BoardType): BoardState {
    const config = BOARD_CONFIGS[boardType];
    return {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: config.size,
      type: boardType,
    };
  }
  private createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    playerNumber: number,
    count: number = 1
  ): BoardState {
    return createHypotheticalBoardWithPlacement(board, position, playerNumber, count);
  }

  private hasAnyLegalMoveOrCaptureFrom(
    from: Position,
    playerNumber: number,
    board: BoardState
  ): boolean {
    const view: PlacementBoardView = {
      isValidPosition: (pos) => this.isValidPosition(pos),
      isCollapsedSpace: (pos, b) => this.isCollapsedSpace(pos, b),
      getMarkerOwner: (pos, b) => this.getMarkerOwner(pos, b),
    };

    return hasAnyLegalMoveOrCaptureFrom(this.gameState.boardType, board, from, playerNumber, view);
  }

  private getPlayerStacks(
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): RingStack[] {
    const stacks: RingStack[] = [];
    for (const stack of board.stacks.values()) {
      if (stack.controllingPlayer === playerNumber) {
        stacks.push(stack);
      }
    }
    return stacks;
  }

  private enumerateCaptureSegmentsFrom(
    from: Position,
    playerNumber: number
  ): Array<{ from: Position; target: Position; landing: Position }> {
    const board = this.gameState.board;

    const adapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, b: BoardState) => this.isCollapsedSpace(pos, b),
      getMarkerOwner: (pos: Position, b: BoardState) => this.getMarkerOwner(pos, b),
    };

    return enumerateCaptureSegmentsFromBoard(
      this.gameState.boardType,
      board,
      from,
      playerNumber,
      adapters
    );
  }

  private applyCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): void {
    const board = this.gameState.board;
    const landingKey = positionToString(landing);

    const landingMarkerOwner = this.getMarkerOwner(landing, board);
    const landedOnOwnMarker = landingMarkerOwner === playerNumber;

    const adapters: CaptureApplyAdapters = {
      applyMarkerEffectsAlongPath: (f, t, player, options) =>
        this.applyMarkerEffectsAlongPath(f, t, player, options),
    };

    applyCaptureSegmentOnBoard(board, from, target, landing, playerNumber, adapters);

    const stacksAfterCapture: Map<string, RingStack> = new Map(board.stacks);
    let eliminatedRingsMap = board.eliminatedRings;
    const state = this.gameState;
    let playersAfterCapture = state.players;
    let totalRingsEliminatedDelta = 0;

    if (landedOnOwnMarker) {
      const stackAtLanding = stacksAfterCapture.get(landingKey);
      if (stackAtLanding && stackAtLanding.stackHeight > 0) {
        const [, ...remainingRings] = stackAtLanding.rings;

        if (remainingRings.length > 0) {
          const newStack: RingStack = {
            ...stackAtLanding,
            rings: remainingRings,
            stackHeight: remainingRings.length,
            capHeight: calculateCapHeight(remainingRings),
            controllingPlayer: remainingRings[0],
          };
          stacksAfterCapture.set(landingKey, newStack);
        } else {
          stacksAfterCapture.delete(landingKey);
        }

        const creditedPlayer = playerNumber;
        eliminatedRingsMap = {
          ...eliminatedRingsMap,
          [creditedPlayer]: (eliminatedRingsMap[creditedPlayer] || 0) + 1,
        };
        playersAfterCapture = playersAfterCapture.map((p) =>
          p.playerNumber === creditedPlayer ? { ...p, eliminatedRings: p.eliminatedRings + 1 } : p
        );
        totalRingsEliminatedDelta = 1;
      }
    }

    const nextState: GameState = {
      ...state,
      board: {
        ...board,
        stacks: stacksAfterCapture,
        markers: new Map(board.markers),
        collapsedSpaces: new Map(board.collapsedSpaces),
        eliminatedRings: eliminatedRingsMap,
      },
      players: playersAfterCapture,
      totalRingsEliminated: state.totalRingsEliminated + totalRingsEliminatedDelta,
    };

    this.gameState = nextState;
  }

  /**
   * Test-only helper: perform a concrete capture chain starting from a
   * specified initial segment using the shared sandbox movement engine.
   * This is used by landing-on-own-marker tests so they can exercise the
   * same overtaking semantics without going through click selection.
   */
  private async performCaptureChain(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): Promise<void> {
    const hooks: SandboxMovementEngineHooks = {
      getGameState: () => this.gameState,
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, board?: BoardState) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos: Position, board?: BoardState) => this.getMarkerOwner(pos, board),
      getPlayerStacks: (p: number, board: BoardState) => this.getPlayerStacks(p, board),
      setMarker: (pos: Position, p: number, board: BoardState) => this.setMarker(pos, p, board),
      collapseMarker: (pos: Position, p: number, board: BoardState) =>
        this.collapseMarker(pos, p, board),
      flipMarker: (pos: Position, p: number, board: BoardState) => this.flipMarker(pos, p, board),
      onMovementComplete: async () => {
        await this.advanceAfterMovement();
      },
    };

    await performCaptureChainSandbox(hooks, from, target, landing, playerNumber);
  }

  private createSandboxTurnHooks(): SandboxTurnHooks {
    return {
      enumerateLegalRingPlacements: (state, playerNumber) =>
        this.enumerateLegalRingPlacements(playerNumber),
      hasAnyLegalMoveOrCaptureFrom: (state, from, playerNumber, board) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      getPlayerStacks: (state, playerNumber, board) => this.getPlayerStacks(playerNumber, board),
      forceEliminateCap: (state, playerNumber) => {
        // forceEliminateCap mutates this.gameState; adapt to functional
        // style by operating on the provided state so callers can treat
        // the hook as a pure function of its arguments.
        this.gameState = state;
        this.forceEliminateCap(playerNumber);
        return this.gameState;
      },
      checkAndApplyVictory: (state) => {
        this.gameState = state;
        this.checkAndApplyVictory();
        return this.gameState;
      },
    };
  }

  private startTurnForCurrentPlayer(): void {
    const hooks = this.createSandboxTurnHooks();

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const { state, turnState } = startTurnForCurrentPlayerSandbox(
      this.gameState,
      turnStateBefore,
      hooks
    );

    this.gameState = state;
    this._hasPlacedThisTurn = turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = turnState.mustMoveFromStackKey;

    this.handleStartOfInteractiveTurn();
  }

  private maybeProcessForcedEliminationForCurrentPlayer(): boolean {
    const hooks = this.createSandboxTurnHooks();

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const result = maybeProcessForcedEliminationForCurrentPlayerSandbox(
      this.gameState,
      turnStateBefore,
      hooks
    );

    this.gameState = result.state;
    this._hasPlacedThisTurn = result.turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = result.turnState.mustMoveFromStackKey;

    return result.eliminated;
  }

  private advanceTurnAndPhaseForCurrentPlayer(): void {
    const hooks = this.createSandboxTurnHooks();

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const { state, turnState } = advanceTurnAndPhaseForCurrentPlayerSandbox(
      this.gameState,
      turnStateBefore,
      hooks
    );

    this.gameState = state;
    this._hasPlacedThisTurn = turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = turnState.mustMoveFromStackKey;

    this.handleStartOfInteractiveTurn();
  }

  private forceEliminateCap(playerNumber: number): void {
    const { board, players } = this.gameState;
    const stacks = this.getPlayerStacks(playerNumber, board);

    const result = forceEliminateCapOnBoard(board, players, playerNumber, stacks);
    if (result.totalRingsEliminatedDelta <= 0) {
      return;
    }

    this.gameState = {
      ...this.gameState,
      board: result.board,
      players: result.players,
      totalRingsEliminated: this.gameState.totalRingsEliminated + result.totalRingsEliminatedDelta,
    };
  }

  private getNextPlayerNumber(current: number): number {
    const players = this.gameState.players;
    const idx = players.findIndex((p) => p.playerNumber === current);
    const nextIdx = (idx + 1) % players.length;
    return players[nextIdx].playerNumber;
  }

  /**
   * Local helper to parse a position string produced by positionToString
   * back into a Position object. This mirrors the backend stringToPosition
   * but is kept local to avoid pulling in additional shared helpers.
   */
  private stringToPositionLocal(posStr: string): Position {
    const parts = posStr.split(',').map(Number);
    if (parts.length === 2) {
      const [x, y] = parts;
      return { x, y };
    }
    if (parts.length === 3) {
      const [x, y, z] = parts;
      return { x, y, z };
    }
    // Defensive fallback; should not occur if positionToString format is
    // consistent.
    return { x: 0, y: 0 };
  }

  /**
   * Local position validity check mirroring BoardManager semantics so we can
   * safely use shared capture helpers on the client.
   */
  private isValidPosition(pos: Position): boolean {
    const config = BOARD_CONFIGS[this.gameState.boardType];
    if (this.gameState.boardType === 'hexagonal') {
      const radius = config.size - 1;
      const x = pos.x;
      const y = pos.y;
      const z = pos.z !== undefined ? pos.z : -x - y;
      const distance = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
      return distance <= radius;
    }
    // Square boards: 0..size-1 grid
    return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
  }

  private isCollapsedSpace(position: Position, board: BoardState = this.gameState.board): boolean {
    const key = positionToString(position);
    return board.collapsedSpaces.has(key);
  }

  private getMarkerOwner(
    position: Position,
    board: BoardState = this.gameState.board
  ): number | undefined {
    const key = positionToString(position);
    const marker = board.markers.get(key);
    return marker?.player;
  }

  private setMarker(
    position: Position,
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): void {
    const key = positionToString(position);

    // Mirror backend BoardManager.setMarker semantics:
    // - Do not place markers on collapsed territory.
    // - Ensure stack+marker exclusivity by removing any stack at this key
    //   before writing the marker.
    if (board.collapsedSpaces.has(key)) {
      return;
    }

    if (board.stacks.has(key)) {
      board.stacks.delete(key);
    }

    board.markers.set(key, {
      player: playerNumber,
      position,
      type: 'regular',
    });
  }

  private flipMarker(
    position: Position,
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): void {
    const key = positionToString(position);
    const existing = board.markers.get(key);
    if (existing && existing.player !== playerNumber) {
      board.markers.set(key, {
        player: playerNumber,
        position,
        type: 'regular',
      });
    }
  }

  private collapseMarker(
    position: Position,
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): void {
    const key = positionToString(position);
    // When a marker collapses to territory, the cell becomes
    // exclusive territory: no stacks or markers may remain.
    board.markers.delete(key);
    board.stacks.delete(key);
    board.collapsedSpaces.set(key, playerNumber);
  }

  /**
   * Find all marker lines on the board for all players. Mirrors
   * BoardManager.findAllLines; only returns lines of at least the
   * configured minimum length.
   *
   * This is primarily used by test-only harnesses (e.g.
   * ClientSandboxEngine.lines.test.ts) via an `any`-cast of the engine
   * instance, so it intentionally remains a private helper rather than
   * part of the public API surface.
   */
  private findAllLines(board: BoardState): LineInfo[] {
    return findAllLinesOnBoard(
      this.gameState.boardType,
      board,
      (pos: Position) => this.isValidPosition(pos),
      (posStr: string) => this.stringToPositionLocal(posStr)
    );
  }

  /**
   * Collapse all markers in `positions` to the given player's territory,
   * removing any stacks there and updating the player's territorySpaces
   * counter. Analogue of GameEngine.collapseLineMarkers.
   */
  private collapseLineMarkers(positions: Position[], playerNumber: number): void {
    const board = this.gameState.board;
    const collapsedKeys = new Set<string>();

    for (const pos of positions) {
      const key = positionToString(pos);
      collapsedKeys.add(key);
      board.markers.delete(key);
      board.stacks.delete(key);
      board.collapsedSpaces.set(key, playerNumber);
    }

    const territoryGain = collapsedKeys.size;
    const updatedPlayers = this.gameState.players.map((p) =>
      p.playerNumber === playerNumber
        ? { ...p, territorySpaces: p.territorySpaces + territoryGain }
        : p
    );

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        markers: new Map(board.markers),
        stacks: new Map(board.stacks),
        collapsedSpaces: new Map(board.collapsedSpaces),
      },
      players: updatedPlayers,
    };
  }

  /**
   * Apply marker effects for a move or capture segment from `from` to `to`:
   * - Leave a marker on the departure space.
   * - For intermediate spaces:
   *   - Opponent markers flip to the mover's color.
   *   - Own markers collapse into territory.
   * - On landing, remove same-color marker (cannot coexist with a stack).
   *
   * The optional `options` parameter is forwarded to the shared
   * applyMarkerEffectsAlongPathOnBoard helper so that callers (notably
   * overtaking captures) can opt out of placing a departure marker on
   * intermediate stacks such as the capture target. This keeps sandbox
   * marker-path semantics aligned with the backend GameEngine for both
   * movement and capture legs.
   */
  private applyMarkerEffectsAlongPath(
    from: Position,
    to: Position,
    playerNumber: number,
    options?: { leaveDepartureMarker?: boolean }
  ): void {
    const board = this.gameState.board;

    const helpers: MarkerPathHelpers = {
      setMarker: (pos, player, b) => this.setMarker(pos, player, b),
      collapseMarker: (pos, player, b) => this.collapseMarker(pos, player, b),
      flipMarker: (pos, player, b) => this.flipMarker(pos, player, b),
    };

    applyMarkerEffectsAlongPathOnBoard(board, from, to, playerNumber, helpers, options);
  }

  // Removed unused handleRingPlacementClick helper to fix TS6133

  /**
   * Helper method to prompt for capture direction when multiple capture
   * options are available for the same landing position.
   */
  private async promptForCaptureDirection(captureOptions: Move[]): Promise<Move> {
    if (captureOptions.length <= 1) {
      return captureOptions[0];
    }

    const choice = {
      id: `sandbox-capture-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      gameId: this.gameState.id,
      playerNumber: this.gameState.currentPlayer,
      type: 'capture_direction' as const,
      prompt: 'Select capture direction',
      options: captureOptions.map((move) => ({
        targetPosition: move.captureTarget!,
        landingPosition: move.to!,
        capturedCapHeight:
          this.gameState.board.stacks.get(positionToString(move.captureTarget!))?.capHeight ?? 0,
      })),
    };

    const response = await this.interactionHandler.requestChoice(choice as any);

    // Find the matching move based on response
    const selected = captureOptions.find((move) => {
      const opt = (response as any).selectedOption;
      return (
        opt &&
        positionToString(opt.targetPosition) === positionToString(move.captureTarget!) &&
        positionToString(opt.landingPosition) === positionToString(move.to!)
      );
    });

    return selected ?? captureOptions[0];
  }

  private async handleMovementClick(position: Position): Promise<void> {
    const board = this.gameState.board;
    const key = positionToString(position);
    const stackAtPos = board.stacks.get(key);

    // Synchronous selection / deselection logic to preserve existing
    // click-to-select semantics used by tests and the UI.
    if (!this.isValidPosition(position)) {
      this._selectedStackKey = undefined;
      return;
    }

    if (!this._selectedStackKey) {
      // If clicking on a stack belonging to the current player, select it.
      if (stackAtPos && stackAtPos.controllingPlayer === this.gameState.currentPlayer) {
        this._selectedStackKey = key;
      }
      return;
    }

    // Clicking the same cell clears selection.
    if (key === this._selectedStackKey) {
      this._selectedStackKey = undefined;
      return;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Adapter-based movement (when orchestrator adapter is enabled)
    // ═══════════════════════════════════════════════════════════════════════
    if (this.useOrchestratorAdapter) {
      const adapter = this.getOrchestratorAdapter();
      const validMoves = adapter.getValidMoves();

      // Find moves from selected stack to clicked position
      const matchingMoves = validMoves.filter(
        (m) =>
          m.from &&
          positionToString(m.from) === this._selectedStackKey &&
          m.to &&
          positionToString(m.to) === key
      );

      if (matchingMoves.length === 0) {
        // No valid move to this position - leave selection unchanged
        return;
      }

      let moveToApply: Move = matchingMoves[0];

      // Handle multiple capture options (same from/to but different targets)
      if (matchingMoves.length > 1) {
        const captureOptions = matchingMoves.filter((m) => m.captureTarget);
        if (captureOptions.length > 1) {
          moveToApply = await this.promptForCaptureDirection(captureOptions);
        }
      }

      // Apply via applyCanonicalMove which handles history recording
      await this.applyCanonicalMove(moveToApply);
      this._selectedStackKey = undefined;
      return;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Legacy movement (when orchestrator adapter is disabled)
    // ═══════════════════════════════════════════════════════════════════════
    const isCanonicalReplay = this._movementInvocationContext === 'canonical';

    // Delegate actual movement / capture application to the shared
    // sandboxMovementEngine, wiring hooks back into this engine.
    const hooks: SandboxMovementEngineHooks = {
      getGameState: () => this.gameState,
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, board?: BoardState) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos: Position, board?: BoardState) => this.getMarkerOwner(pos, board),
      getPlayerStacks: (playerNumber: number, board: BoardState) =>
        this.getPlayerStacks(playerNumber, board),
      setMarker: (pos: Position, playerNumber: number, board: BoardState) =>
        this.setMarker(pos, playerNumber, board),
      collapseMarker: (pos: Position, playerNumber: number, board: BoardState) =>
        this.collapseMarker(pos, playerNumber, board),
      flipMarker: (pos: Position, playerNumber: number, board: BoardState) =>
        this.flipMarker(pos, playerNumber, board),
      chooseCaptureSegment: async (
        options: Array<{ from: Position; target: Position; landing: Position }>
      ) => {
        if (options.length <= 1) {
          return options[0];
        }

        const playerNumber = this.gameState.currentPlayer;
        const choice = {
          id: `sandbox-capture-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          gameId: this.gameState.id,
          playerNumber,
          type: 'capture_direction' as const,
          prompt: 'Select capture direction',
          options: options.map((opt) => ({
            targetPosition: opt.target,
            landingPosition: opt.landing,
            capturedCapHeight:
              this.gameState.board.stacks.get(positionToString(opt.target))?.capHeight ?? 0,
          })),
        };

        const response = await this.interactionHandler.requestChoice(choice as any);
        const selected = options.find((opt) => {
          const o = (response as any).selectedOption;
          return (
            o &&
            positionToString(o.targetPosition) === positionToString(opt.target) &&
            positionToString(o.landingPosition) === positionToString(opt.landing)
          );
        });

        return selected ?? options[0];
      },
      // For human-driven movement, record canonical history for both
      // capture segments and simple moves via the movement engine hooks.
      ...(isCanonicalReplay
        ? {}
        : {
            onCaptureSegmentApplied: (info: any) => this.handleCaptureSegmentApplied(info),
            onSimpleMoveApplied: (info: any) => this.handleSimpleMoveApplied(info),
          }),
      onMovementComplete: async () => {
        await this.advanceAfterMovement();
      },
    };

    const result = await handleMovementClickSandbox(hooks, this._selectedStackKey, position);
    this._selectedStackKey = result.nextSelectedFromKey;
  }

  // Removed unused advanceAfterPlacement helper to fix TS6133

  private async advanceAfterMovement(): Promise<void> {
    this.debugCheckpoint('before-advanceAfterMovement');

    // Post-movement consequences for the player who just moved: lines,
    // territory disconnections, and victory checks.
    await this.processLinesForCurrentPlayer();
    this.debugCheckpoint('after-processLinesForCurrentPlayer');

    // When in traceMode and a decision phase was set, stop here so the
    // current player remains active to make the decision move on their
    // next AI turn. This keeps sandbox traces aligned with backend
    // move-driven decision phases.
    if (this.gameState.currentPhase === 'line_processing') {
      if (
        typeof process !== 'undefined' &&
        (process as any).env &&
        (process as any).env.NODE_ENV === 'test'
      ) {
        // eslint-disable-next-line no-console
        console.log('[ClientSandboxEngine.advanceAfterMovement] EARLY RETURN: line_processing');
      }
      return;
    }

    await this.processDisconnectedRegionsForCurrentPlayer();
    this.debugCheckpoint('after-processDisconnectedRegionsForCurrentPlayer');

    // Same traceMode handling for territory_processing
    if (this.gameState.currentPhase === 'territory_processing') {
      if (
        typeof process !== 'undefined' &&
        (process as any).env &&
        (process as any).env.NODE_ENV === 'test'
      ) {
        // eslint-disable-next-line no-console
        console.log(
          '[ClientSandboxEngine.advanceAfterMovement] EARLY RETURN: territory_processing',
          'currentPlayer=',
          this.gameState.currentPlayer
        );
      }
      return;
    }

    this.checkAndApplyVictory();
    this.debugCheckpoint('after-checkAndApplyVictory');

    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    // Hand off to the shared turn/phase sequencer so that sandbox turn
    // rotation and forced elimination mirror backend semantics. By
    // normalising the phase to territory_processing here we are telling
    // the shared helper that all automatic bookkeeping for this player
    // (lines and territory) has completed.
    this.gameState = {
      ...this.gameState,
      currentPhase: 'territory_processing',
    };

    if (
      typeof process !== 'undefined' &&
      (process as any).env &&
      (process as any).env.NODE_ENV === 'test'
    ) {
      // eslint-disable-next-line no-console
      console.log(
        '[ClientSandboxEngine.advanceAfterMovement] Calling advanceTurnAndPhaseForCurrentPlayer',
        'currentPlayer=',
        this.gameState.currentPlayer
      );
    }

    this.advanceTurnAndPhaseForCurrentPlayer();
    this.debugCheckpoint('after-advanceTurnAndPhaseForCurrentPlayer');
  }

  /**
   * Process all disconnected regions for the current player using the
   * sandboxTerritory engine helper. This mirrors the backend GameEngine
   * behaviour, including RegionOrderChoice handling when multiple eligible
   * regions exist, while keeping the implementation purely functional.
   */
  private async processDisconnectedRegionsForCurrentPlayer(): Promise<void> {
    // Guard: when exactly one player has stacks on the board, there is no
    // meaningful notion of a "disconnected" region for self-elimination
    // purposes. The backend territory processor is only exercised in
    // practice once multiple players have on-board presence; without this
    // guard the sandbox can incorrectly treat an early sparse position
    // (e.g. after Player 1's very first move in a mixed human/AI game) as
    // a fully disconnected region and immediately collapse the entire
    // board to territory, triggering an early victory.
    const activePlayers = new Set<number>();
    for (const stack of this.gameState.board.stacks.values()) {
      activePlayers.add(stack.controllingPlayer);
    }
    // Mirror backend territory-processing semantics: only short-circuit when
    // the *moving player* is the sole player with stacks on the board.
    // When exactly one player has stacks but that player is not the mover,
    // we must still consult findDisconnectedRegions so that scenarios like
    // FAQ Q23 (control without on-board presence) are handled consistently
    // across engines.
    if (activePlayers.size === 1 && activePlayers.has(this.gameState.currentPlayer)) {
      return;
    }

    if (this.traceMode) {
      const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
      const eligible = disconnected.filter((region) =>
        this.canProcessDisconnectedRegion(
          region.spaces,
          this.gameState.currentPlayer,
          this.gameState.board
        )
      );

      if (eligible.length > 0) {
        this.gameState = {
          ...this.gameState,
          currentPhase: 'territory_processing',
        };
        return;
      }
    }

    const { state: nextState, pendingSelfElimination } =
      await processDisconnectedRegionsForCurrentPlayerEngine(
        this.gameState,
        this.interactionHandler as unknown as TerritoryInteractionHandler,
        (regionSpaces: Position[], playerNumber: number, state: GameState) =>
          this.canProcessDisconnectedRegion(regionSpaces, playerNumber, state.board)
      );

    this.gameState = nextState;

    // The engine helper now performs in-loop self-elimination per region
    // (matching backend behavior), so we only need to set the flag for
    // traceMode callers that want explicit elimination decisions.
    // Note: In traceMode, we return early above before calling the helper,
    // so this flag setting is primarily for future extensibility.
    if (pendingSelfElimination && this.traceMode) {
      this._pendingTerritorySelfElimination = true;
    }
  }

  /**
   * Self-elimination prerequisite: the current player must have at least
   * one stack outside the disconnected region before it can be processed.
   */
  private canProcessDisconnectedRegion(
    regionSpaces: Position[],
    playerNumber: number,
    board: BoardState
  ): boolean {
    // Thin wrapper around the shared self-elimination prerequisite helper so
    // sandbox territory gating stays aligned with backend GameEngine /
    // RuleEngine semantics. We construct a transient Territory wrapper here
    // purely for gating; controllingPlayer is not inspected by the helper.
    const region: Territory = {
      spaces: regionSpaces,
      controllingPlayer: playerNumber,
      isDisconnected: true,
    };

    return canProcessTerritoryRegion(board, region, { player: playerNumber });
  }

  /**
   * Enumerate canonical territory-processing decision Moves for the
   * current player in the sandbox. This now delegates region discovery
   * and Q23 gating to the shared + sandbox territory helpers so that
   * backend GameEngine, RuleEngine, and sandbox observe an identical
   * decision surface:
   *
   * - process_territory_region: choose which eligible disconnected
   *   region to process first, subject to the self-elimination
   *   prerequisite from §12.2 / FAQ Q23.
   *
   * For normal human-driven sandbox games, territory collapse continues
   * to be driven automatically from advanceAfterMovement; this helper is
   * intended for canonical replay and advanced parity harnesses.
   */
  private getValidTerritoryProcessingMovesForCurrentPlayer(): Move[] {
    return getValidTerritoryProcessingMoves(
      this.gameState,
      (regionSpaces: Position[], playerNumber: number, state: GameState) =>
        this.canProcessDisconnectedRegion(regionSpaces, playerNumber, state.board)
    );
  }

  /**
   * Enumerate explicit self-elimination decision Moves for the current
   * player in the sandbox. This mirrors the backend
   * RuleEngine.getValidEliminationDecisionMoves helper and is primarily
   * used by parity/debug tooling:
   *
   * - eliminate_rings_from_stack: choose which controlled stack/cap to
   *   self-eliminate from when an elimination is required.
   *
   * For now, this helper does not alter sandbox turn flow; it simply
   * exposes the canonical decision surface so tests and future UI/AI
   * can treat elimination as a Move selection problem.
   *
   * Enumeration itself is delegated to the shared
   * {@link enumerateTerritoryEliminationMoves} helper so that backend and
   * sandbox share identical elimination options and diagnostics; local
   * pending flags only control *when* these options are surfaced.
   */
  private getValidEliminationDecisionMovesForCurrentPlayer(): Move[] {
    const pendingTerritory = this._pendingTerritorySelfElimination;
    const pendingLineReward = this._pendingLineRewardElimination;

    // Explicit elimination decisions are only legal when a self-elimination
    // debt is outstanding from either a prior territory decision or a
    // line-reward decision, mirroring the backend GameEngine flags.
    if (!pendingTerritory && !pendingLineReward) {
      return [];
    }

    const movingPlayer = this.gameState.currentPlayer;
    return enumerateTerritoryEliminationMoves(this.gameState, movingPlayer);
  }

  /**
   * Apply ring-elimination and territory-control victory checks after
   * post-movement processing. When a winner is found, the sandbox game
   * is marked as completed and subsequent moves are ignored.
   */
  private checkAndApplyVictory(): void {
    const hooks: SandboxGameEndHooks = {
      enumerateLegalRingPlacements: (playerNumber: number) =>
        this.enumerateLegalRingPlacements(playerNumber),
    };

    // Delegate stalemate resolution + victory detection to the shared
    // sandbox game-end helpers so that semantics stay aligned across
    // hosts. This mirrors the backend GameEngine.checkGameEnd flow.
    const before = this.gameState;
    const { state, result } = checkAndApplyVictorySandbox(this.gameState, hooks);

    // Test-only diagnostic logging: when a victory is detected, emit a
    // compact snapshot so we can understand why gameStatus flipped to
    // 'completed' in early-turn scenarios (e.g. mixedPlayers tests).
    if (
      result &&
      typeof process !== 'undefined' &&
      !!(process as any).env &&
      (process as any).env.NODE_ENV === 'test'
    ) {
      // eslint-disable-next-line no-console
      console.log('[ClientSandboxEngine Victory Debug]', {
        reason: result.reason,
        currentPlayerBefore: before.currentPlayer,
        currentPhaseBefore: before.currentPhase,
        gameStatusBefore: before.gameStatus,
        currentPlayerAfter: state.currentPlayer,
        currentPhaseAfter: state.currentPhase,
        gameStatusAfter: state.gameStatus,
        players: before.players.map((p) => ({
          playerNumber: p.playerNumber,
          type: p.type,
          ringsInHand: p.ringsInHand,
          eliminatedRings: p.eliminatedRings,
          territorySpaces: p.territorySpaces,
        })),
        stacks: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
          key,
          height: stack.stackHeight,
          cap: stack.capHeight,
          rings: stack.rings,
        })),
      });
    }

    this.gameState = state;

    // Normalise terminal states so that completed games are never left in
    // the dedicated line/territory decision phases. This mirrors the backend
    // GameEngine.applyDecisionMove elimination handling and keeps both
    // parity traces and UI-facing games in a stable 'ring_placement' phase
    // after victory.
    if (
      this.gameState.gameStatus !== 'active' &&
      (this.gameState.currentPhase === 'territory_processing' ||
        this.gameState.currentPhase === 'line_processing')
    ) {
      this.gameState = {
        ...this.gameState,
        currentPhase: 'ring_placement',
      };
    }

    if (!result) {
      return;
    }

    this.victoryResult = result;
  }

  /**
   * Detect and process marker lines for the current player.
   *
   * For human-driven games, this now routes line resolution through the
   * same canonical Move helpers used for AI and backend trace replays:
   *
   * - Exact-length lines: apply a `process_line` Move that collapses all
   *   markers in the line and eliminates a cap (via
   *   applyCanonicalProcessLine).
   * - Overlength lines: apply a `process_line` Move that collapses only
   *   the minimum required markers with no elimination, matching the
   *   default sandbox behaviour (no line_reward_option choice yet).
   *
   * This ensures that all line-processing effects are both driven by and
   * recorded as canonical Moves, keeping sandbox history aligned with
   * backend semantics while preserving the existing automatic behaviour
   * when no explicit line-order/reward decisions are exposed in the UI.
   */
  /**
   * Enumerate canonical line-processing decision Moves for the current
   * player in the sandbox.
   *
   * This is now a thin adapter over {@link getValidLineProcessingMoves}
   * from sandboxLinesEngine, which itself delegates geometry and move
   * shapes to the shared {@link enumerateProcessLineMoves} and
   * {@link enumerateChooseLineRewardMoves} helpers in
   * src/shared/engine/lineDecisionHelpers.ts.
   */
  private getValidLineProcessingMovesForCurrentPlayer(): Move[] {
    return getValidLineProcessingMoves(this.gameState);
  }

  private async processLinesForCurrentPlayer(): Promise<void> {
    // Keep applying lines for the current player until none remain.
    // We use the sandboxLinesEngine.getValidLineProcessingMoves helper
    // (which delegates to shared lineDecisionHelpers) to identify
    // candidates and applyLineDecisionMove to execute them.
    //
    // Automatic sandbox behaviour remains:
    // - Exact-length lines: collapse all markers and immediately eliminate
    //   a cap via forceEliminateCapOnBoard.
    // - Overlength lines: default to Option 2 (minimum contiguous subset of
    //   length L, no elimination) for the first available line.
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const moves = getValidLineProcessingMoves(this.gameState);
      const processLineMoves = moves.filter((m) => m.type === 'process_line');

      if (processLineMoves.length === 0) {
        break;
      }

      const boardType = this.gameState.boardType;
      const requiredLength = BOARD_CONFIGS[boardType].lineLength;

      // Default behaviour: pick the first line for the current player.
      let moveToApply: Move = processLineMoves[0];
      const line = moveToApply.formedLines && moveToApply.formedLines[0];
      if (!line) {
        break;
      }

      // For overlength lines, prefer a MINIMUM_COLLAPSE reward choice
      // (length L contiguous subset) rather than the raw process_line
      // decision, to preserve the legacy "Option 2 by default" sandbox
      // behaviour.
      if (line.length > requiredLength) {
        const lineKey = line.positions.map((p) => positionToString(p)).join('|');
        const rewardCandidates = moves.filter(
          (m) =>
            m.type === 'choose_line_reward' &&
            m.formedLines &&
            m.formedLines.length > 0 &&
            m.formedLines[0].positions.length === line.positions.length &&
            m.formedLines[0].positions.map((p) => positionToString(p)).join('|') === lineKey
        );

        const preferredSegment = line.positions.slice(0, requiredLength);
        const preferredKey = preferredSegment.map((p) => positionToString(p)).join('|');

        const minCollapse =
          rewardCandidates.find(
            (m) =>
              m.collapsedMarkers &&
              m.collapsedMarkers.length === requiredLength &&
              m.collapsedMarkers.map((p) => positionToString(p)).join('|') === preferredKey
          ) ||
          rewardCandidates.find(
            (m) => m.collapsedMarkers && m.collapsedMarkers.length === requiredLength
          );

        if (minCollapse) {
          moveToApply = minCollapse;
        }
      }

      if (this.traceMode) {
        // In trace/parity mode we only need to surface that a line decision
        // is available; the concrete decision Moves will be applied via
        // applyCanonicalMove. Normalise the phase and return.
        this.gameState = {
          ...this.gameState,
          currentPhase: 'line_processing',
        };
        return;
      }

      // Capture state before applying the line so history snapshots have
      // correct before/after semantics.
      const beforeState = this.getGameState();

      // Apply the move using the shared helper.
      const outcome = applyLineDecisionMove(this.gameState, moveToApply);
      const nextState = outcome.nextState;

      // Check if state actually changed
      if (hashGameState(nextState) === hashGameState(this.gameState)) {
        break;
      }

      this.gameState = nextState;

      // For automatic sandbox flows, immediately apply the cap elimination
      // when the shared helper reports a pending line-reward elimination
      // (exact-length lines and collapse-all rewards).
      if (outcome.pendingLineRewardElimination) {
        this.forceEliminateCap(moveToApply.player);
      }

      // Record the canonical decision in history so that parity harnesses
      // can replay the exact same sequence into both engines.
      this.appendHistoryEntry(beforeState, moveToApply);
    }
  }

  /**
   * Attempt to place one or more rings for the current player at the given
   * position during the ring_placement phase.
   *
   * Canonical rules enforced:
   * - Never place on collapsed spaces.
   * - Multi-ring placement is only allowed on empty cells.
   * - When placing onto an existing stack, at most one ring is added per
   *   placement action (additional requested rings are ignored).
   * - Resulting stack must have at least one legal move/capture
   *   (no-dead-placement).
   *
   * Returns true if the placement was applied, false otherwise.
   */
  public async tryPlaceRings(position: Position, requestedCount: number): Promise<boolean> {
    if (this.gameState.gameStatus !== 'active') {
      return false;
    }

    if (this.gameState.currentPhase !== 'ring_placement') {
      return false;
    }

    if (!this.isValidPosition(position)) {
      return false;
    }

    const board = this.gameState.board;
    const key = positionToString(position);

    // Do not allow placement on collapsed territory.
    if (board.collapsedSpaces.has(key)) {
      return false;
    }

    // Do not allow placement on markers; stacks and markers must never
    // coexist on the same space. This mirrors backend RuleEngine
    // semantics and keeps the S-invariant aligned between engines.
    if (board.markers.has(key)) {
      return false;
    }

    const player = this.gameState.players.find(
      (p) => p.playerNumber === this.gameState.currentPlayer
    );
    if (!player || player.ringsInHand <= 0) {
      return false;
    }

    const existingStack = board.stacks.get(key);
    const isOccupied = !!existingStack && existingStack.rings.length > 0;
    const maxFromHand = player.ringsInHand;

    // Canonical rule: only 1 ring per placement onto an existing stack.
    const maxPerPlacement = isOccupied ? 1 : maxFromHand;
    const effectiveCount = Math.min(Math.max(requestedCount, 1), maxPerPlacement);

    if (effectiveCount <= 0) {
      return false;
    }

    // No-dead-placement: after placing effectiveCount rings here, the resulting
    // stack must have at least one legal move/capture.
    const hypotheticalBoard = this.createHypotheticalBoardWithPlacement(
      board,
      position,
      this.gameState.currentPlayer,
      effectiveCount
    );

    if (
      !this.hasAnyLegalMoveOrCaptureFrom(position, this.gameState.currentPlayer, hypotheticalBoard)
    ) {
      return false;
    }

    const nextStacks = new Map(board.stacks);

    if (isOccupied && existingStack) {
      const addedRings = Array(effectiveCount).fill(this.gameState.currentPlayer);
      const rings = [...addedRings, ...existingStack.rings];
      const newStack: RingStack = {
        ...existingStack,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: this.gameState.currentPlayer,
      };
      nextStacks.set(key, newStack);
    } else {
      const rings = Array(effectiveCount).fill(this.gameState.currentPlayer);
      const newStack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: this.gameState.currentPlayer,
      };
      nextStacks.set(key, newStack);
    }

    const updatedPlayers = this.gameState.players.map((p) =>
      p.playerNumber === this.gameState.currentPlayer
        ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - effectiveCount) }
        : p
    );

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: nextStacks,
      },
      players: updatedPlayers,
      currentPhase: 'movement',
    };

    this._hasPlacedThisTurn = true;
    this._mustMoveFromStackKey = key;
    this._selectedStackKey = key;

    // Process lines immediately after placement, mirroring backend GameEngine
    // behaviour where processAutomaticConsequences runs after every move.
    // This ensures that if a placement completes a line, it is collapsed
    // (and potentially eliminates the placed stack) before the movement phase.
    await this.processLinesForCurrentPlayer();
    if (this.gameState.currentPhase === 'line_processing') {
      return true;
    }

    await this.processDisconnectedRegionsForCurrentPlayer();
    if (this.gameState.currentPhase === 'territory_processing') {
      return true;
    }

    this.checkAndApplyVictory();

    return true;
  }

  /**
   * Internal canonical move-applier used by both AI turns and
   * applyCanonicalMove. It mutates this.gameState according to the given
   * Move and returns true when the move was applied and changed state.
   *
   * When useOrchestratorAdapter is enabled, this method delegates to the
   * shared orchestrator via SandboxOrchestratorAdapter, falling back to
   * legacy sandbox logic only when the adapter is disabled.
   */
  private async applyCanonicalMoveInternal(
    move: Move,
    opts: { bypassNoDeadPlacement?: boolean } = {}
  ): Promise<boolean> {
    this.debugCheckpoint(`before-applyCanonicalMoveInternal-${move.type}`);
    const beforeHash = hashGameState(this.getGameState());

    // ═══════════════════════════════════════════════════════════════════════
    // Orchestrator Adapter Delegation
    // ═══════════════════════════════════════════════════════════════════════
    // When the orchestrator adapter is enabled, delegate all rules logic
    // to the shared orchestrator. This eliminates duplicated logic and
    // ensures sandbox and backend use identical rules processing.
    if (this.useOrchestratorAdapter) {
      const beforeState = this.getGameState();
      const changed = await this.processMoveViaAdapter(move, beforeState);

      // Debug checkpoint after adapter processing
      this.debugCheckpoint(`after-applyCanonicalMoveInternal-adapter-${move.type}`);

      // Verify state actually changed
      const afterHash = hashGameState(this.getGameState());
      return changed && beforeHash !== afterHash;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Legacy Sandbox Logic (when adapter is disabled)
    // ═══════════════════════════════════════════════════════════════════════

    // Ensure currentPlayer matches the move's player for the purposes of
    // canonical application.
    if (move.player !== this.gameState.currentPlayer) {
      this.gameState = {
        ...this.gameState,
        currentPlayer: move.player,
      };
    }

    let applied = false;

    switch (move.type) {
      case 'place_ring': {
        if (!move.to) {
          break;
        }
        if (opts.bypassNoDeadPlacement) {
          // Backend-style placement: bypass no-dead-placement gating and
          // directly add rings at the destination, clearing any marker.
          const board = this.gameState.board;
          const key = positionToString(move.to);
          const placementCount = Math.max(1, move.placementCount ?? 1);

          board.markers.delete(key);

          const existingStack = board.stacks.get(key);
          const placementRings = new Array(placementCount).fill(move.player);

          let newRings: number[];
          if (existingStack && existingStack.rings.length > 0) {
            newRings = [...placementRings, ...existingStack.rings];
          } else {
            newRings = placementRings;
          }

          const newStack: RingStack = {
            position: move.to,
            rings: newRings,
            stackHeight: newRings.length,
            capHeight: calculateCapHeight(newRings),
            controllingPlayer: newRings[0],
          };

          const nextStacks = new Map(board.stacks);
          nextStacks.set(key, newStack);

          const updatedPlayers = this.gameState.players.map((p) =>
            p.playerNumber === move.player
              ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - placementCount) }
              : p
          );

          this.gameState = {
            ...this.gameState,
            board: {
              ...board,
              stacks: nextStacks,
            },
            players: updatedPlayers,
            currentPhase: 'movement',
          };

          // When applying canonical placement moves (either from the
          // sandbox AI or backend trace replays), enforce the same
          // must-move semantics as the interactive sandbox: the next
          // movement action for this player must originate from the
          // just-updated stack. This keeps movement reachability and
          // valid-move enumeration aligned with the backend
          // RuleEngine/GameEngine, which only exposes moves from the
          // placed stack immediately after placement.
          this._hasPlacedThisTurn = true;
          this._mustMoveFromStackKey = key;
          this._selectedStackKey = key;

          applied = true;
        } else {
          // Sandbox-style placement: enforce no-dead-placement via
          // tryPlaceRings so AI turns share the same gating as human
          // sandbox interaction.
          const count = Math.max(1, move.placementCount ?? 1);
          const placed = await this.tryPlaceRings(move.to, count);
          applied = placed;
        }
        break;
      }

      case 'skip_placement': {
        this.gameState = {
          ...this.gameState,
          currentPhase: 'movement',
        };
        applied = true;
        break;
      }

      case 'move_ring':
      case 'move_stack': {
        if (!move.from || !move.to) {
          break;
        }
        // Reuse existing movement handler by simulating a source
        // selection followed by a destination click. Mark this invocation
        // as canonical so movement hooks do not emit duplicate history.
        this._movementInvocationContext = 'canonical';
        try {
          this._selectedStackKey = positionToString(move.from);
          await this.handleMovementClick(move.to);
        } finally {
          this._movementInvocationContext = null;
        }
        // handleMovementClick calls advanceAfterMovement internally via hooks,
        // so we don't need to call it again here.
        applied = true;
        break;
      }

      case 'overtaking_capture':
      case 'continue_capture_segment': {
        if (!move.from || !move.to || !move.captureTarget) {
          break;
        }

        // Apply the segment using the same helper used by sandbox-driven chains.
        this.applyCaptureSegment(move.from, move.captureTarget, move.to, move.player);

        // After applying the segment, determine whether the chain can continue
        // from the new landing position under the current board state.
        const continuationOptions = this.enumerateCaptureSegmentsFrom(move.to, move.player);

        if (continuationOptions.length > 0) {
          // At least one follow-up capture segment is available. Mirror backend
          // behaviour by entering the interactive 'chain_capture' phase while
          // deferring automatic post-movement consequences until the chain
          // eventually terminates.
          if (this.gameState.currentPhase !== 'chain_capture') {
            this.gameState = {
              ...this.gameState,
              currentPhase: 'chain_capture',
            };
          }
        } else {
          // No legal continuations remain; the capture chain is complete.
          // Process lines and territory consequences but do NOT advance the
          // turn yet - that happens AFTER appendHistoryEntry in applyCanonicalMove
          // to match backend GameEngine ordering where history is recorded BEFORE
          // advanceGame().
          await this.processLinesForCurrentPlayer();
          if (this.gameState.currentPhase === 'line_processing') {
            applied = true;
            break;
          }

          await this.processDisconnectedRegionsForCurrentPlayer();
          if (this.gameState.currentPhase === 'territory_processing') {
            applied = true;
            break;
          }

          this.checkAndApplyVictory();

          // Set phase to territory_processing so the caller knows to advance
          // the turn after recording history.
          if (this.gameState.gameStatus === 'active') {
            this.gameState = {
              ...this.gameState,
              currentPhase: 'territory_processing',
            };
          }
        }

        applied = true;
        break;
      }

      case 'process_line':
      case 'choose_line_reward': {
        // Use shared helpers for line decision application.
        // This unifies traceMode and normal mode logic for applying the move.
        const outcome =
          move.type === 'process_line'
            ? applyProcessLineDecision(this.gameState, move)
            : applyChooseLineRewardDecision(this.gameState, move);

        const nextState = outcome.nextState;

        if (hashGameState(nextState) !== hashGameState(this.gameState)) {
          this.gameState = nextState;
          applied = true;

          // Update pending elimination flag based on the outcome.
          if (outcome.pendingLineRewardElimination) {
            this._pendingLineRewardElimination = true;
          }

          // In traceMode, we stop here and let the AI/harness drive the next step.
          if (this.traceMode) {
            this.gameState = {
              ...this.gameState,
              currentPhase: 'line_processing',
            };
          } else {
            // In normal mode, continue processing lines or advance.
            const remainingLines = this.findAllLines(this.gameState.board).filter(
              (line) => line.player === move.player
            );

            if (remainingLines.length > 0) {
              this.gameState = {
                ...this.gameState,
                currentPhase: 'line_processing',
              };
            } else {
              // Check for territory processing
              const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
              const eligible = disconnected.filter((region) =>
                this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
              );

              if (eligible.length > 0) {
                this.gameState = {
                  ...this.gameState,
                  currentPhase: 'territory_processing',
                };
              } else {
                this.checkAndApplyVictory();
                if (this.gameState.gameStatus === 'active') {
                  this._hasPlacedThisTurn = false;
                  this._mustMoveFromStackKey = undefined;
                  const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                  this.gameState = {
                    ...this.gameState,
                    currentPlayer: nextPlayer,
                  };
                  this.startTurnForCurrentPlayer();
                }
              }
            }
          }
        }
        break;
      }

      case 'process_territory_region': {
        // Use shared helper for territory region application.
        const outcome = applyProcessTerritoryRegionDecision(this.gameState, move);
        const nextState = outcome.nextState;

        if (hashGameState(nextState) !== hashGameState(this.gameState)) {
          this.gameState = nextState;
          applied = true;

          // Update pending elimination flag based on the outcome.
          if (outcome.pendingSelfElimination) {
            this._pendingTerritorySelfElimination = true;
          }

          if (this.traceMode) {
            // In trace/parity mode, run victory checks immediately but stay in
            // territory_processing so explicit elimination can follow.
            this.checkAndApplyVictory();
            if (this.gameState.gameStatus === 'active') {
              this.gameState = {
                ...this.gameState,
                currentPhase: 'territory_processing',
              };
            }
          } else {
            // In normal mode, check if more regions remain or advance.
            const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
            const eligible = disconnected.filter((region) =>
              this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
            );

            if (eligible.length === 0) {
              this.checkAndApplyVictory();
              if (this.gameState.gameStatus === 'active') {
                this._hasPlacedThisTurn = false;
                this._mustMoveFromStackKey = undefined;
                const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                this.gameState = {
                  ...this.gameState,
                  currentPlayer: nextPlayer,
                };
                this.startTurnForCurrentPlayer();
              }
            }
          }
        }
        break;
      }

      case 'eliminate_rings_from_stack': {
        const wasTerritorySelfElimination = this._pendingTerritorySelfElimination;
        const wasLineRewardElimination = this._pendingLineRewardElimination;

        // Use shared helper for elimination application.
        const { nextState } = applyEliminateRingsFromStackDecision(this.gameState, move);

        if (hashGameState(nextState) !== hashGameState(this.gameState)) {
          this.gameState = nextState;
          applied = true;

          // Clear pending flags.
          this._pendingTerritorySelfElimination = false;
          this._pendingLineRewardElimination = false;

          if (wasLineRewardElimination) {
            // Line-reward elimination complete.
            this.checkAndApplyVictory();

            if (this.gameState.gameStatus === 'active') {
              const remainingLines = this.findAllLines(this.gameState.board).filter(
                (line) => line.player === move.player
              );

              if (remainingLines.length > 0) {
                this.gameState = {
                  ...this.gameState,
                  currentPhase: 'line_processing',
                };
              } else {
                this.gameState = {
                  ...this.gameState,
                  currentPhase: 'territory_processing',
                };
              }
            }
          } else if (wasTerritorySelfElimination) {
            // Territory-origin self-elimination.
            const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
            const eligible = disconnected.filter((region) =>
              this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
            );

            if (eligible.length > 0) {
              this.gameState = {
                ...this.gameState,
                currentPhase: 'territory_processing',
              };
            } else {
              this.checkAndApplyVictory();
              if (this.gameState.gameStatus === 'active') {
                this._hasPlacedThisTurn = false;
                this._mustMoveFromStackKey = undefined;
                const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                this.gameState = {
                  ...this.gameState,
                  currentPlayer: nextPlayer,
                };
                this.startTurnForCurrentPlayer();
              }
            }
          } else {
            // Fallback / generic elimination.
            const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
            const eligible = disconnected.filter((region) =>
              this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
            );

            if (eligible.length === 0) {
              this.checkAndApplyVictory();
              if (this.gameState.gameStatus === 'active') {
                this._hasPlacedThisTurn = false;
                this._mustMoveFromStackKey = undefined;
                const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                this.gameState = {
                  ...this.gameState,
                  currentPlayer: nextPlayer,
                };
                this.startTurnForCurrentPlayer();
              }
            }
          }
        }
        break;
      }

      default: {
        // Unsupported move types are treated as no-ops here; callers that
        // care about strictness (e.g. applyCanonicalMove) can enforce
        // additional checks around this helper.
        break;
      }
    }

    if (!applied) {
      return false;
    }

    this.debugCheckpoint(`after-applyCanonicalMoveInternal-${move.type}`);
    const afterHash = hashGameState(this.getGameState());
    const changed = beforeHash !== afterHash;

    // Test-only board invariant enforcement: when running under Jest, assert
    // that the sandbox never commits a board state with overlapping stacks,
    // markers, or collapsed spaces. This mirrors the backend
    // BoardManager.assertBoardInvariants helper but is intentionally wired
    // only for tests so production builds avoid the extra scan cost.
    if (
      changed &&
      typeof process !== 'undefined' &&
      (process as any).env &&
      (process as any).env.NODE_ENV === 'test'
    ) {
      const selfAny = this as any;
      if (typeof selfAny.assertBoardInvariants === 'function') {
        selfAny.assertBoardInvariants(`applyCanonicalMoveInternal:${move.type}`);
      }
    }

    return changed;
  }

  /**
   * Test-only helper: apply a single process_territory_region Move using the
   * same canonical pipeline as applyCanonicalMove, returning a boolean that
   * indicates whether the move changed state. This exists so RulesMatrix
   * territory scenarios can exercise Q23 preconditions against the sandbox
   * without going through the full turn/phase machinery.
   */
  private async applyCanonicalProcessTerritoryRegion(move: Move): Promise<boolean> {
    if (move.type !== 'process_territory_region') {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalProcessTerritoryRegion: expected process_territory_region, got ${
          (move as any).type
        }`
      );
    }

    if (this.gameState.gameStatus !== 'active') {
      return false;
    }

    return this.applyCanonicalMoveInternal(move, {
      bypassNoDeadPlacement: true,
    });
  }

  /**
   * Test-only helper: apply a backend-style Move into the sandbox engine.
   *
   * This is used by parallel debug harnesses to replay the same canonical
   * move sequence into both engines. It intentionally bypasses the sandbox
   * AI heuristics and, for placement, the no-dead-placement gating so that
   * we can mirror backend GameEngine behaviour as closely as possible.
   */
  public async applyCanonicalMove(move: Move): Promise<void> {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const beforeStateForHistory = this.getGameState();

    const supportedTypes: Move['type'][] = [
      'place_ring',
      'skip_placement',
      'move_ring',
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
      'process_line',
      'choose_line_reward',
      'process_territory_region',
      'eliminate_rings_from_stack',
    ];

    if (!supportedTypes.includes(move.type)) {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalMove: unsupported move type ${(move as any).type}`
      );
    }

    const changed = await this.applyCanonicalMoveInternal(move, {
      bypassNoDeadPlacement: true,
    });

    if (changed) {
      // For capture completions (when phase is territory_processing after
      // applyCanonicalMoveInternal ran line/territory processing), advance
      // the turn BEFORE recording history to prepare for the next player.
      // This matches backend behavior where advanceGame() runs before
      // appendHistoryEntry(), so phaseAfter captures the next player's turn.
      //
      // IMPORTANT: Only advance if there are NO pending territory decisions.
      // If eligible disconnected regions exist, we must wait for explicit
      // process_territory_region moves before advancing.
      if (
        this.gameState.gameStatus === 'active' &&
        this.gameState.currentPhase === 'territory_processing' &&
        (move.type === 'overtaking_capture' || move.type === 'continue_capture_segment')
      ) {
        // Check for pending territory regions before advancing
        const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
        const eligible = disconnected.filter((region) =>
          this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
        );

        // Only advance if no eligible regions are pending
        if (eligible.length === 0) {
          this.advanceTurnAndPhaseForCurrentPlayer();
        }
      }

      // Record history AFTER advancing the turn for captures.
      // This matches backend ordering where history is recorded after advanceGame().
      this.appendHistoryEntry(beforeStateForHistory, move);
    }
  }
}

// Test-only: attach a lightweight board-invariant helper to the prototype so
// invariant tests can exercise internal board sanity checks without expanding
// the public class surface for production code.
(ClientSandboxEngine.prototype as any).assertBoardInvariants = function (
  this: ClientSandboxEngine,
  context: string
): void {
  const isTestEnv =
    typeof process !== 'undefined' &&
    !!(process as any).env &&
    (process as any).env.NODE_ENV === 'test';

  const board: BoardState = (this as any).gameState.board as BoardState;
  const errors: string[] = [];

  // Invariant 1: no stacks may exist on collapsed territory.
  for (const key of board.stacks.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`stack present on collapsed space at ${key}`);
    }
  }

  // Invariant 2: a cell may not host both a stack and a marker.
  for (const key of board.stacks.keys()) {
    if (board.markers.has(key)) {
      errors.push(`stack and marker coexist at ${key}`);
    }
  }

  // Invariant 3: a cell may not host both a marker and collapsed territory.
  for (const key of board.markers.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`marker present on collapsed space at ${key}`);
    }
  }

  if (errors.length === 0) {
    return;
  }

  const message =
    `ClientSandboxEngine invariant violation (${context}):` + '\n' + errors.join('\n');

  // eslint-disable-next-line no-console
  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
};
