/**
 * @fileoverview Client Sandbox Engine - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It bridges client UI interactions (clicks, decisions) with the canonical rules.
 *
 * Canonical SSoT:
 * - Rules/engine: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Types: `src/shared/types/game.ts`
 * - FSM: `src/shared/engine/fsm/TurnStateMachine.ts`
 *
 * This adapter:
 * - Translates UI events (clicks, selections) to canonical Move operations
 * - Manages local state for offline/sandbox play
 * - Delegates ALL rules logic to the canonical shared engine
 * - Handles sandbox-specific UX (history playback, scenario loading, AI turns)
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 * Per RR-CANON-R070, FSM validation is the canonical move validator.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
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
  LineRewardChoice,
  RegionOrderChoice,
  Territory,
  MarkerPathHelpers,
  LocalAIRng,
  PendingDecision,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  stringToPosition,
  positionsEqual,
  hashGameState,
  isValidPosition,
  playerHasMaterial,
  canProcessTerritoryRegion,
  enumerateProcessTerritoryRegionMoves,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  getEffectiveLineLengthThreshold,
  findLinesForPlayer,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
  enumerateTerritoryEliminationMoves,
  applyCaptureSegment as applyCaptureSegmentAggregate,
  enumerateAllCaptureMoves as enumerateAllCaptureMovesAggregate,
  computeRingEliminationVictoryThreshold,
  // Canonical placement aggregate API (TS SSOT)
  enumeratePlacementPositions,
  validatePlacementAggregate,
  applyPlacementMoveAggregate,
  // Type guards for move narrowing
  isCaptureMove,
  type PlaceRingAction,
  // LPS tracking helpers
  createLpsTrackingState,
  updateLpsTracking,
  evaluateLpsVictory,
  buildLpsVictoryResult,
  isLpsActivePhase,
  // Swap sides (pie rule) helpers
  shouldOfferSwapSides,
  // Player state helpers
  hasAnyRealAction,
  // Forced elimination helpers
  enumerateForcedEliminationOptions,
  applyForcedEliminationForPlayer,
  // Global action helpers
  playerHasAnyRings,
} from '../../shared/engine';
import { normalizeLegacyMoveType } from '../../shared/engine/legacy/legacyMoveTypes';
import type { LpsTrackingState } from '../../shared/engine';
import {
  deserializeGameState,
  type SerializedGameState,
} from '../../shared/engine/contracts/serialization';
// Phase 1 decomposition: History and State managers
import {
  type HistoryManagerHooks,
  appendHistoryEntry as appendHistoryEntryFromManager,
  recordHistorySnapshotsOnly as recordHistorySnapshotsOnlyFromManager,
  getStateAtMoveIndex as getStateAtMoveIndexFromManager,
  rebuildSnapshotsFromMoveHistory as rebuildSnapshotsFromMoveHistoryFromManager,
} from './sandboxHistoryManager';
import { getSerializedState } from './sandboxStateManager';
// Shared action-availability predicates (canonical implementations)
import {
  hasAnyPlacementForPlayer,
  hasAnyMovementForPlayer,
  hasAnyCaptureForPlayer,
} from '../../shared/engine';
import type {
  PlayerChoice,
  PlayerChoiceResponseFor,
  PlayerChoiceResponse,
  CaptureDirectionChoice,
  RingEliminationChoice,
} from '../../shared/types/game';
import {
  isSandboxAiTraceModeEnabled,
  isSandboxAiStallDiagnosticsEnabled,
  isSandboxLpsDebugEnabled,
  isTestEnvironment,
  debugLog,
} from '../../shared/utils/envFlags';
import { SeededRNG, generateGameSeed } from '../../shared/utils/rng';
import { applyMarkerEffectsAlongPathOnBoard } from '../../shared/engine';
import {
  enumerateSimpleMovementLandings,
  isPlayerEligibleForRecovery,
  enumerateRecoverySlideLandings,
} from './sandboxMovement';
import { findAllLinesOnBoard } from './sandboxLines';
import { findDisconnectedRegionsOnBoard } from './sandboxTerritory';
import { forceEliminateCapOnBoard } from './sandboxElimination';
import { SandboxGameEndHooks, checkAndApplyVictorySandbox } from './sandboxGameEnd';
import type { PerTurnState as SandboxTurnState, TurnLogicDelegates } from '../../shared/engine';
import { advanceTurnAndPhase } from '../../shared/engine';
import {
  createHypotheticalBoardWithPlacement,
  hasAnyLegalMoveOrCaptureFrom,
  PlacementBoardView,
} from './sandboxPlacement';
import { maybeRunAITurnSandbox, SandboxAIHooks } from './sandboxAI';
import {
  SandboxOrchestratorAdapter,
  SandboxStateAccessor,
  SandboxDecisionHandler,
} from './SandboxOrchestratorAdapter';
import {
  mapPendingDecisionToPlayerChoice as mapDecisionToChoice,
  mapPlayerChoiceResponseToMove as mapResponseToMove,
  buildCaptureDirectionChoice,
  type DecisionMappingContext,
} from './sandboxDecisionMapping';

const SANDBOX_AI_STALL_DIAGNOSTICS = isSandboxAiStallDiagnosticsEnabled();
const normalizeMoveType = (moveType: Move['type']): Move['type'] =>
  normalizeLegacyMoveType(moveType);

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
 *
 * It is an orchestration/UX host over the shared engine and must not
 * introduce new rules semantics. For allowed rules surfaces, see
 * `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md` (§0 Rules Entry Surfaces).
 */

export type SandboxPlayerKind = PlayerType; // 'human' | 'ai'

export interface SandboxConfig {
  boardType: BoardType;
  numPlayers: number;
  playerKinds: SandboxPlayerKind[]; // indexed 0..3 for players 1..4
  /** AI difficulty levels per player (1-10), indexed 0..3 for players 1..4 */
  aiDifficulties?: number[];
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

/**
 * Test-only augmented interface for ClientSandboxEngine with board invariant
 * assertions. This method is attached to the prototype only in test environments.
 */
interface ClientSandboxEngineTestAugmented {
  assertBoardInvariants(context: string): void;
}

export class ClientSandboxEngine {
  private gameState: GameState;
  private interactionHandler: SandboxInteractionHandler;
  // When true, the engine is running under a trace/replay harness. This
  // is currently reserved for future parity-specific behaviour and does
  // not alter normal sandbox rules or AI policy.
  private readonly traceMode: boolean;
  // AI difficulty levels per player (1-10), indexed 0..3 for players 1..4
  private readonly aiDifficulties: number[];

  // ═══════════════════════════════════════════════════════════════════════
  // Orchestrator Adapter Integration
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Orchestrator adapter for turn processing.
   *
   * PERMANENTLY ENABLED as of 2025-12-01 (Phase 3 migration complete).
   * The orchestrator is now the canonical turn processor. Legacy path removed.
   *
   * The orchestrator adapter provides a single executable rules implementation
   * derived from the canonical rules spec (`RULES_CANONICAL_SPEC.md`) via the
   * shared TS engine, eliminating duplicated logic between client/server.
   *
   * @see docs/archive/ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md
   */

  /** Lazily-initialized adapter instance */
  private orchestratorAdapter: SandboxOrchestratorAdapter | null = null;

  // Per-game RNG for deterministic AI behavior
  private rng: SeededRNG;

  // When non-null, the sandbox game has ended with this result.
  private victoryResult: GameResult | null = null;

  // Canonical explanation for why the game ended, when available.
  private gameEndExplanation: GameEndExplanation | null = null;

  // Internal turn-level state for sandbox per-turn flow.
  private _hasPlacedThisTurn: boolean = false;
  private _mustMoveFromStackKey: string | undefined;
  // Track rings placed this turn for the 3-ring-per-turn limit
  private _ringsPlacedThisTurn: number = 0;
  // Track the position where rings were placed this turn (all must go to same position)
  private _placementPositionThisTurn: string | undefined;

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

  // Transient highlight buffer used by the sandbox host to render a brief
  // visual cue for newly-collapsed line segments. Populated by
  // processLinesForCurrentPlayer and consumed via consumeRecentLineHighlights.
  private _recentLineHighlightKeys: string[] = [];

  // State snapshots for history playback. Each entry is a deep clone of the
  // game state after the corresponding move was applied.
  // _stateSnapshots[0] = state after move 1, _stateSnapshots[N-1] = state after move N
  // For moveIndex=0 (initial state), use _initialStateSnapshot.
  private _stateSnapshots: GameState[] = [];
  // The initial game state before any moves were applied (for moveIndex=0).
  private _initialStateSnapshot: GameState | null = null;

  // Test-only checkpoint hook used by parity/diagnostic harnesses to capture
  // GameState snapshots at key points inside canonical move application and
  // post-movement processing. When unset, all debugCheckpoint calls are no-ops.
  /**
   * Host-internal metadata for last-player-standing (R172) detection.
   * This uses the shared LpsTrackingState from lpsTracking.ts to track,
   * per round, which players had any real actions available (placement,
   * non-capture movement, or overtaking capture) at the start of their
   * most recent interactive turn. Kept off of GameState so sandbox
   * snapshots and wire formats remain unchanged.
   */
  private _lpsState: LpsTrackingState = createLpsTrackingState();

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
   *
   * @param skipMoveHistory - When true, skip updating moveHistory (use when
   *   the orchestrator adapter has already added the move to moveHistory).
   */
  private appendHistoryEntry(
    before: GameState,
    action: Move,
    opts?: { skipMoveHistory?: boolean }
  ): void {
    // Delegate to the extracted history manager
    appendHistoryEntryFromManager(this.getHistoryManagerHooks(), before, action, opts);
  }

  /**
   * Record history playback snapshots for an action that was already appended
   * to `moveHistory`/`history` by the SandboxOrchestratorAdapter.
   *
   * This avoids double-recording GameHistoryEntry rows (which inflates
   * `historyLength` and produces odd/even moveNumber gaps in exported fixtures),
   * while still keeping `_stateSnapshots` aligned for HistoryPlayback UX.
   */
  private recordHistorySnapshotsOnly(before: GameState): void {
    // Delegate to the extracted history manager
    recordHistorySnapshotsOnlyFromManager(this.getHistoryManagerHooks(), before);
  }

  constructor(opts: ClientSandboxEngineOptions) {
    const { config, interactionHandler, traceMode } = opts;
    this.interactionHandler = interactionHandler;
    this.traceMode = traceMode ?? isSandboxAiTraceModeEnabled();

    // Initialize with temporary seed; will be set from gameState if provided
    this.rng = new SeededRNG(generateGameSeed());

    const board = this.createEmptyBoard(config.boardType);
    const clampDifficulty = (value: number) => Math.max(1, Math.min(10, Math.round(value)));

    // Store AI difficulties for use by sandbox AI hooks (default to D4 if not provided)
    this.aiDifficulties = Array.from({ length: 4 }, (_, idx) =>
      clampDifficulty(config.aiDifficulties?.[idx] ?? 4)
    );

    const players: Player[] = Array.from({ length: config.numPlayers }, (_, idx) => {
      const playerNumber = idx + 1;
      const kind = config.playerKinds[idx] ?? 'human';
      const aiDifficulty = this.aiDifficulties[idx];
      return {
        id: `sandbox-${playerNumber}`,
        username: `Player ${playerNumber}`,
        type: kind,
        playerNumber,
        isReady: true,
        timeRemaining: 0,
        ...(kind === 'ai' ? { aiDifficulty } : {}),
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
      // Pie rule (swap sides) is opt-in for 2-player games.
      // Data shows P2 wins >55% with pie rule enabled by default.
      // For 3p/4p games the flag is omitted and swap_sides is never surfaced.
      ...(config.numPlayers === 2 ? { rulesOptions: { swapRuleEnabled: false } } : {}),
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
      // Per RR-CANON-R061: victoryThreshold = round((2/3) × ownStartingRings + (1/3) × opponentsCombinedStartingRings)
      // Simplified: round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
      victoryThreshold: computeRingEliminationVictoryThreshold(
        boardConfig.ringsPerPlayer,
        config.numPlayers
      ),
      territoryVictoryThreshold: Math.floor(boardConfig.totalSpaces / 2) + 1,
    };

    // Initialize orchestrator adapter lazily when first needed
    this.orchestratorAdapter = null;
  }

  /**
   * Create a HistoryManagerHooks interface for interacting with the history manager.
   * This provides the hooks pattern interface for state access without circular dependencies.
   */
  private getHistoryManagerHooks(): HistoryManagerHooks {
    return {
      getGameState: () => this.getGameState(),
      updateGameState: (state: GameState) => {
        this.gameState = state;
      },
      getStateSnapshots: () => this._stateSnapshots,
      setStateSnapshots: (snapshots: GameState[]) => {
        this._stateSnapshots = snapshots;
      },
      getInitialStateSnapshot: () => this._initialStateSnapshot,
      setInitialStateSnapshot: (snapshot: GameState | null) => {
        this._initialStateSnapshot = snapshot;
      },
    };
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
          return {
            type: 'ai',
            ...(player.aiDifficulty !== undefined ? { aiDifficulty: player.aiDifficulty } : {}),
          };
        }
        return { type: 'human' };
      },
    };

    const decisionHandler: SandboxDecisionHandler = {
      requestDecision: async (decision) => {
        // For region_order decisions, populate board.territories with the region
        // geometry so that:
        // 1. gameViewModels.ts can highlight all cells in each region
        // 2. useSandboxInteractions.ts click handler can match clicks to regions
        // The regionId keys must match those used in mapPendingDecisionToPlayerChoice
        // which uses `region-${idx}` for non-skip options.
        if (decision.type === 'region_order') {
          const state = this.gameState;
          const newTerritories = new Map(state.board.territories);
          decision.options.forEach((opt, idx) => {
            if (opt.type === 'skip_territory_processing') {
              // Skip options don't have geometry to display
              return;
            }
            const region = opt.disconnectedRegions?.[0];
            if (region) {
              // Use `region-${idx}` to match the regionId format in sandboxDecisionMapping.ts
              newTerritories.set(`region-${idx}`, region);
            }
          });
          this.gameState = {
            ...state,
            board: {
              ...state.board,
              territories: newTerritories,
            },
          };
        }

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
        ...(this._debugCheckpointHook && {
          debugHook: (label: string, state: GameState) => {
            this._debugCheckpointHook?.(label, state);
          },
        }),
        onError: (error: Error, context: string) => {
          console.error(`[SandboxOrchestratorAdapter] Error in ${context}:`, error);
        },
      },
      // In traceMode (replay), skip auto-resolving territory decisions so explicit
      // choose_territory_option moves from the recording are used instead (legacy alias: process_territory_region).
      skipTerritoryAutoResolve: this.traceMode,
    });
  }

  /**
   * Map a PendingDecision from the orchestrator to a PlayerChoice for
   * the sandbox interaction handler.
   */
  private mapPendingDecisionToPlayerChoice(decision: PendingDecision): PlayerChoice {
    // Delegate to extracted pure function for testability
    const context: DecisionMappingContext = {
      gameId: this.gameState.id,
      board: this.gameState.board,
    };
    return mapDecisionToChoice(decision, context);
  }

  /**
   * Map a PlayerChoice response back to a Move for the orchestrator.
   */
  private mapPlayerChoiceResponseToMove(
    decision: PendingDecision,
    response: PlayerChoiceResponse<unknown> & {
      selectedLineIndex?: number;
      selectedRegionIndex?: number;
    }
  ): Move {
    // Delegate to extracted pure function for testability
    return mapResponseToMove(decision, response);
  }

  /**
   * Process a move via the orchestrator adapter.
   *
   * This method delegates to the shared orchestrator for rules logic,
   * handling sandbox-specific concerns like history recording.
   */
  private async processMoveViaAdapter(
    move: Move,
    _beforeStateForHistory: GameState
  ): Promise<boolean> {
    const adapter = this.getOrchestratorAdapter();

    // Diagnostic logging for AI stall investigation
    const stateBefore = adapter.getGameState();

    debugLog(SANDBOX_AI_STALL_DIAGNOSTICS, '[processMoveViaAdapter] Before processMove:', {
      moveType: move.type,
      movePlayer: move.player,
      moveFrom: move.from ? positionToString(move.from) : null,
      moveTo: move.to ? positionToString(move.to) : null,
      stateCurrentPlayer: stateBefore.currentPlayer,
      stateCurrentPhase: stateBefore.currentPhase,
      stateGameStatus: stateBefore.gameStatus,
    });

    const result = await adapter.processMove(move);

    debugLog(SANDBOX_AI_STALL_DIAGNOSTICS, '[processMoveViaAdapter] After processMove:', {
      success: result.success,
      error: result.error,
      stateChanged: result.metadata?.stateChanged,
      nextPhase: result.nextState.currentPhase,
      nextPlayer: result.nextState.currentPlayer,
    });

    if (!result.success) {
      const message = result.error || 'Orchestrator processMove failed';
      // Surface orchestrator errors as hard failures so that canonical
      // replay / parity harnesses treat mis-phased or otherwise invalid
      // moves as structural issues, matching Python’s strict semantics.
      throw new Error(
        `[SandboxOrchestratorAdapter] processMove failed for move type '${move.type}': ${message}`
      );
    }

    // Update victory result if game ended
    if (result.victoryResult) {
      this.victoryResult = result.victoryResult;
    }
    this.gameEndExplanation = result.gameEndExplanation ?? null;

    const canonicalType = normalizeMoveType(move.type);

    // Update internal turn state based on move type
    if (canonicalType === 'place_ring' && move.to) {
      this._hasPlacedThisTurn = true;
      this._mustMoveFromStackKey = positionToString(move.to);
      this._selectedStackKey = positionToString(move.to);
    } else if (
      canonicalType === 'move_stack' ||
      canonicalType === 'overtaking_capture' ||
      canonicalType === 'continue_capture_segment'
    ) {
      // RR-CANON-R093: Track the landing position so subsequent captures in this
      // turn are restricted to originate from the stack that just moved.
      if (move.to) {
        const toKey = positionToString(move.to);
        const fromKey = move.from ? positionToString(move.from) : undefined;
        // If mustMoveFromStackKey was set (from placement), only update if
        // this move originates from that stack. Otherwise, always set it to
        // the landing position to enforce capture eligibility.
        if (!this._mustMoveFromStackKey || fromKey === this._mustMoveFromStackKey) {
          this._mustMoveFromStackKey = toKey;
        }
      }
      this._selectedStackKey = undefined;
    }

    // Clear pending flags based on move type
    if (move.type === 'eliminate_rings_from_stack') {
      this._pendingLineRewardElimination = false;
      this._pendingTerritorySelfElimination = false;
    }

    // Sync internal turn state when orchestrator advances to a new player's turn.
    // The orchestrator handles turn advancement internally, but the sandbox's
    // internal flags need to be reset to match the new player's ring_placement phase.
    const currentState = this.getGameState();
    if (
      currentState.currentPhase === 'ring_placement' &&
      currentState.currentPlayer !== move.player
    ) {
      // Turn has advanced to the next player's ring_placement phase
      this._hasPlacedThisTurn = false;
      this._mustMoveFromStackKey = undefined;
      this._selectedStackKey = undefined;
      this._ringsPlacedThisTurn = 0;
      this._placementPositionThisTurn = undefined;
      this.handleStartOfInteractiveTurn();
    }

    // Enforce board invariants after state updates via orchestrator adapter.
    // This ensures the same S-invariant checking that the legacy path performs
    // in applyCanonicalMoveInternal after line 2972.
    const stateChanged = result.metadata?.stateChanged ?? true;
    if (stateChanged && isTestEnvironment()) {
      const testAugmented = this as unknown as ClientSandboxEngineTestAugmented;
      if (typeof testAugmented.assertBoardInvariants === 'function') {
        testAugmented.assertBoardInvariants(`processMoveViaAdapter:${move.type}`);
      }
    }

    return stateChanged;
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
  /**
   * Get the AI difficulty level for a specific player (1-10).
   * Returns undefined for human players or if not set.
   */
  public getAIDifficulty(playerNumber: number): number | undefined {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.type !== 'ai') {
      return undefined;
    }
    // Use stored difficulties array (0-indexed)
    return this.aiDifficulties[playerNumber - 1];
  }

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
   * When non-null, contains the canonical explanation for why the game ended.
   * This is used by HUD/Victory surfaces to render concept-aligned copy.
   */
  public getGameEndExplanation(): GameEndExplanation | null {
    return this.gameEndExplanation;
  }

  /**
   * Get the current Last-Player-Standing (LPS) tracking state.
   * Used by UI to display LPS round counter and progress toward LPS victory.
   * Per RR-CANON-R172, LPS requires 3 consecutive rounds where only 1 player has real actions.
   */
  public getLpsTrackingState(): {
    roundIndex: number;
    consecutiveExclusiveRounds: number;
    consecutiveExclusivePlayer: number | null;
    exclusivePlayerForCompletedRound: number | null;
  } {
    return {
      roundIndex: this._lpsState.roundIndex,
      consecutiveExclusiveRounds: this._lpsState.consecutiveExclusiveRounds,
      consecutiveExclusivePlayer: this._lpsState.consecutiveExclusivePlayer,
      exclusivePlayerForCompletedRound: this._lpsState.exclusivePlayerForCompletedRound,
    };
  }

  /**
   * Sandbox-only helper: return and clear the most recently-collapsed line
   * positions for the current game. The sandbox host uses this to render a
   * brief visual cue after automatic line processing, without embedding any
   * rules logic in React.
   */
  public consumeRecentLineHighlights(): Position[] {
    const keys = this._recentLineHighlightKeys;
    this._recentLineHighlightKeys = [];
    return keys.map((posStr) => stringToPosition(posStr));
  }

  /**
   * Get a serialized snapshot of the current game state.
   * Used for saving custom scenarios.
   */
  public getSerializedState(): SerializedGameState {
    return getSerializedState(this.getGameState());
  }

  /**
   * Get the game state at a specific move index for history playback.
   *
   * @param moveIndex - The move index to retrieve state for:
   *   - 0 = initial state (before any moves)
   *   - N = state after move N (1-indexed moves)
   *   - Total moves = current/final state
   * @returns The GameState at that point, or null if unavailable
   *
   * This method extracts state from the history entries which contain
   * before/after snapshots for each move. For fixtures loaded with
   * pre-existing move history, historical states are only available
   * if the history entries contain the snapshots.
   */
  public getStateAtMoveIndex(moveIndex: number): GameState | null {
    // Delegate to the extracted history manager
    return getStateAtMoveIndexFromManager(this.getHistoryManagerHooks(), moveIndex);
  }

  /**
   * Rebuild state snapshots from a game's move history.
   *
   * When loading a fixture/saved state that has move history, this method
   * reconstructs the intermediate game states by replaying all moves from
   * a fresh initial state. This enables history playback for loaded games.
   *
   * @param finalState - The final game state (after all moves)
   */
  private rebuildSnapshotsFromMoveHistory(finalState: GameState): void {
    // Delegate to the extracted history manager
    rebuildSnapshotsFromMoveHistoryFromManager(this.getHistoryManagerHooks(), finalState);
  }

  /**
   * Initialize the sandbox engine from a pre-existing serialized game state.
   * Used by the Scenario Picker to load test vectors and saved states.
   *
   * This replaces the current game state with the deserialized state and
   * resets all internal tracking flags to match a fresh game from that point.
   *
   * @param serializedState - The serialized game state to load
   * @param playerKinds - Player types for each seat (human/ai)
   * @param interactionHandler - Handler for player decisions
   */
  public initFromSerializedState(
    serializedState: SerializedGameState,
    playerKinds: SandboxPlayerKind[],
    interactionHandler: SandboxInteractionHandler,
    aiDifficulties?: number[]
  ): void {
    // 1. Deserialize the game state using existing utility
    let gameState = deserializeGameState(serializedState);

    // 1a. Normalise structurally completed but mid-capture states that may
    // have been produced by older sandbox AI stall behaviour or imported
    // ringrift_sandbox_fixture_v1 snapshots. Completed games should not
    // present as if they were still in a capture/decision phase.
    if (gameState.gameStatus === 'completed') {
      const hasLegacyMustMoveCursor =
        (gameState as unknown as { mustMoveFromStackKey?: string | undefined })
          .mustMoveFromStackKey !== undefined;

      gameState = {
        ...gameState,
        currentPhase: 'ring_placement',
        chainCapturePosition: undefined,
        ...(hasLegacyMustMoveCursor ? { mustMoveFromStackKey: undefined } : {}),
      } as GameState;
    }

    // 1b. Clear stale chainCapturePosition for active games when not in chain_capture phase.
    // Saved states may have inconsistent chainCapturePosition values that were never cleared.
    if (
      gameState.gameStatus === 'active' &&
      gameState.currentPhase !== 'chain_capture' &&
      gameState.chainCapturePosition !== undefined
    ) {
      gameState = {
        ...gameState,
        chainCapturePosition: undefined,
      } as GameState;
    }

    // 2. Apply player types to the deserialized state
    // Jan 10, 2026: Default to max difficulty (10) for optimal AI play
    const clampDifficulty = (value: number) => Math.max(1, Math.min(10, Math.round(value)));
    gameState.players = gameState.players.map((p, idx) => {
      const kind = playerKinds[idx] ?? 'human';
      if (kind === 'ai') {
        const desired = aiDifficulties?.[idx] ?? p.aiDifficulty ?? 10;
        return {
          ...p,
          type: kind,
          aiDifficulty: clampDifficulty(desired),
        };
      }

      return {
        ...p,
        type: kind,
        aiDifficulty: undefined,
      };
    });

    // 3. Update the interaction handler
    this.interactionHandler = interactionHandler;

    // 4. Reset internal per-turn state flags
    // RR-FIX-2026-01-11: Sync _mustMoveFromStackKey from loaded state instead of
    // resetting to undefined. Fixtures may represent mid-turn states where a ring
    // was placed and movement is constrained to that position.
    const loadedMustMove = gameState.mustMoveFromStackKey;
    this._hasPlacedThisTurn = !!loadedMustMove;
    this._mustMoveFromStackKey = loadedMustMove;
    this._selectedStackKey = loadedMustMove;
    this._ringsPlacedThisTurn = loadedMustMove ? 1 : 0;
    this._placementPositionThisTurn = loadedMustMove;
    this._movementInvocationContext = null;
    this._lastAIMove = null;
    this._pendingLineRewardElimination = false;
    this._pendingTerritorySelfElimination = false;
    // 5. Reset LPS tracking state using shared helper
    this._lpsState = createLpsTrackingState();

    // 5b. Rebuild state snapshots from move history for playback support.
    // When loading a fixture with move history, we reconstruct snapshots by
    // replaying all moves from the initial state.
    this._stateSnapshots = [];
    this._initialStateSnapshot = null;
    this.rebuildSnapshotsFromMoveHistory(gameState);

    // 6. Clear victory result
    this.victoryResult = null;
    this.gameEndExplanation = null;

    // 7. Re-initialize RNG from seed if present, otherwise generate new seed
    const gameSeed = gameState.rngSeed ?? generateGameSeed();
    this.rng = new SeededRNG(gameSeed);

    // Ensure the game state has the seed
    gameState.rngSeed = gameSeed;

    // 8. Set the game state
    this.gameState = gameState;

    // 9. Reset orchestrator adapter to pick up new state
    this.orchestratorAdapter = null;
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
   * Test-only helper: enumerate legal Moves for the current player using
   * sandbox semantics. This is used by orchestrator-vs-legacy parity tests
   * to drive both engines with the same canonical Move sequence.
   */
  public getValidMoves(playerNumber: number): Move[] {
    // Delegate move enumeration to the shared orchestrator so that sandbox AI
    // and parity harnesses see the exact same canonical Move surface as backend hosts.
    const adapter = this.getOrchestratorAdapter();
    const state = this.gameState;
    if (state.currentPlayer !== playerNumber || state.gameStatus !== 'active') {
      return [];
    }
    return adapter.getValidMoves();
  }

  /**
   * Helper for chain-capture UX: derive the current chain origin position and
   * the set of legal continuation landing positions for the current player,
   * based on canonical continue_capture_segment moves from the orchestrator.
   *
   * Returns null when the game is not in chain_capture phase or when no
   * continuation moves exist.
   */
  public getChainCaptureContextForCurrentPlayer(): { from: Position; landings: Position[] } | null {
    const state = this.getGameState();

    // DEBUG: Log chain capture context request
    // eslint-disable-next-line no-console
    console.log('[ClientSandboxEngine.getChainCaptureContextForCurrentPlayer] Called:', {
      gameStatus: state.gameStatus,
      currentPhase: state.currentPhase,
      currentPlayer: state.currentPlayer,
      chainCapturePosition: state.chainCapturePosition,
    });

    if (state.gameStatus !== 'active' || state.currentPhase !== 'chain_capture') {
      // eslint-disable-next-line no-console
      console.log(
        '[ClientSandboxEngine.getChainCaptureContextForCurrentPlayer] Early exit: not in chain_capture phase'
      );
      return null;
    }

    const allMoves = this.getValidMoves(state.currentPlayer);
    // DEBUG: Log all moves
    // eslint-disable-next-line no-console
    console.log(
      '[ClientSandboxEngine.getChainCaptureContextForCurrentPlayer] getValidMoves returned:',
      {
        totalMoves: allMoves.length,
        moveTypes: allMoves.map((m) => m.type),
      }
    );

    const moves = allMoves.filter((m) => m.type === 'continue_capture_segment');

    // DEBUG: Log filtered moves
    // eslint-disable-next-line no-console
    console.log('[ClientSandboxEngine.getChainCaptureContextForCurrentPlayer] After filtering:', {
      continueMoves: moves.length,
      moves: moves.map((m) => ({ type: m.type, from: m.from, to: m.to })),
    });

    if (moves.length === 0) {
      // eslint-disable-next-line no-console
      console.log(
        '[ClientSandboxEngine.getChainCaptureContextForCurrentPlayer] No continuation moves found'
      );
      return null;
    }

    const from = moves[0].from as Position;
    const landings: Position[] = [];
    const seen = new Set<string>();

    for (const move of moves) {
      const to = move.to as Position | undefined;
      if (!to) continue;
      const key = positionToString(to);
      if (seen.has(key)) continue;
      seen.add(key);
      landings.push(to);
    }

    return { from, landings };
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

    const phase = this.gameState.currentPhase;

    if (phase === 'ring_placement') {
      const beforeState = this.getGameState();
      const playerNumber = beforeState.currentPlayer;

      const key = positionToString(pos);
      const existingBefore = beforeState.board.stacks.get(key);

      // If we've already placed rings this turn, check if clicking on a valid
      // landing target to trigger movement
      if (this._ringsPlacedThisTurn > 0 && this._placementPositionThisTurn) {
        const placementPos = stringToPosition(this._placementPositionThisTurn);
        if (!positionsEqual(pos, placementPos)) {
          // Check if this is a valid landing from our placement position
          const landingPositions = this.getValidLandingPositionsForCurrentPlayer(placementPos);
          const isValidLanding = landingPositions.some((t) => positionsEqual(t, pos));

          if (isValidLanding) {
            // User wants to move after placing - transition to movement and apply
            this.gameState = {
              ...this.gameState,
              currentPhase: 'movement',
            };

            // RR-FIX-2026-01-11: Reset _selectedStackKey to placement position before
            // calling handleMovementClick. Exploratory clicks during ring_placement
            // (at line 1188) may have changed _selectedStackKey to a different position,
            // causing handleMovementClick to look for moves from the wrong stack.
            this._selectedStackKey = this._placementPositionThisTurn;

            // Apply the movement move
            this._movementInvocationContext = 'human';
            try {
              await this.handleMovementClick(pos);
            } finally {
              this._movementInvocationContext = null;
            }
            return;
          }
        }
      }

      // RR-FIX-2026-01-11: Handle movement when player has 0 rings in hand.
      // When the player can't place rings (ringsInHand == 0), they're still in
      // ring_placement phase but need to move. Check if they have a stack selected
      // and are clicking on a valid landing position for that stack.
      const player = beforeState.players.find((p) => p.playerNumber === playerNumber);
      const canPlaceRings = player && player.ringsInHand > 0;
      if (!canPlaceRings && this._selectedStackKey) {
        const selectedPos = stringToPosition(this._selectedStackKey);
        const landingPositions = this.getValidLandingPositionsForCurrentPlayer(selectedPos);
        const isValidLanding = landingPositions.some((t) => positionsEqual(t, pos));

        if (isValidLanding) {
          // User wants to move from selected stack - transition to movement and apply
          this.gameState = {
            ...this.gameState,
            currentPhase: 'movement',
          };

          // Apply the movement move
          this._movementInvocationContext = 'human';
          try {
            await this.handleMovementClick(pos);
          } finally {
            this._movementInvocationContext = null;
          }
          return;
        }
      }

      // Preserve placed-on-stack metadata for history, mirroring the backend
      // place_ring representation.
      const placedOnStack = !!existingBefore && existingBefore.rings.length > 0;

      // Check if we've already placed rings this turn at a different position
      // All rings in a turn must go to the same position
      if (this._placementPositionThisTurn && this._placementPositionThisTurn !== key) {
        // Clicking on a different position after placing - just select it (don't place)
        // This allows the user to select a different stack to see its movement options
        this._selectedStackKey = key;
        return;
      }

      // Check if we've reached the 3-ring-per-turn limit
      if (this._ringsPlacedThisTurn >= 3) {
        // Already placed 3 rings, can't place more - just select
        this._selectedStackKey = key;
        return;
      }

      // Selection logic for stacking on existing stacks:
      // - Click on ANY stack (own or opponent) not yet selected → just select it (don't place)
      // - Click on selected stack → place ring on top (taking control if opponent's)
      // - Click on empty space → place ring there
      // Rules §2.1: "On existing stack (any owner)" - placing on opponent stacks takes control
      // Exception: if this is the position we've been placing at, always allow placement
      if (existingBefore && this._placementPositionThisTurn !== key) {
        // Clicking on any existing stack that's not our placement position
        if (this._selectedStackKey !== key) {
          // Not currently selected → select it and return (don't place yet)
          this._selectedStackKey = key;
          return;
        }
        // Already selected → fall through to place ring on top
      }

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

      // Track placement for 3-ring-per-turn limit
      this._ringsPlacedThisTurn += 1;
      this._placementPositionThisTurn = key;

      // After successful placement, set _selectedStackKey so clicking the same
      // cell again will place another ring (instead of just selecting it).
      this._selectedStackKey = key;

      // Mark that we've placed this turn and set the must-move key
      this._hasPlacedThisTurn = true;
      this._mustMoveFromStackKey = key;

      // Check if player still has rings available for additional placements
      const currentPlayer = this.gameState.players.find((p) => p.playerNumber === playerNumber);
      const hasRingsRemaining = currentPlayer && currentPlayer.ringsInHand > 0;

      // Keep phase as ring_placement if we can still place more rings (up to 3)
      // AND the player has rings remaining in hand.
      // The adapter may have transitioned to 'movement', but we override that
      // so humans can continue placing rings with subsequent clicks.
      if (this._ringsPlacedThisTurn < 3 && hasRingsRemaining) {
        this.gameState = {
          ...this.gameState,
          currentPhase: 'ring_placement',
        };
      } else if (
        !hasRingsRemaining &&
        this._mustMoveFromStackKey &&
        this.gameState.currentPhase === 'ring_placement'
      ) {
        // RR-FIX-2026-01-12: Player placed their last ring and has no rings remaining.
        // Auto-advance to movement phase so they can move from the placed stack.
        // Emit no_placement_action to properly advance the phase via the orchestrator.
        // Only do this if we're still in ring_placement phase (orchestrator may have
        // already advanced the phase).
        const advanceMoveNumber = this.gameState.history.length + 1;
        const noPlacementMove: Move = {
          type: 'no_placement_action',
          player: playerNumber,
          id: `no-placement-action-auto-${advanceMoveNumber}`,
          moveNumber: advanceMoveNumber,
          timestamp: new Date(),
          thinkTime: 0,
          to: { x: 0, y: 0 },
        } as Move;
        await this.applyCanonicalMove(noPlacementMove);
      }

      // The orchestrator adapter already recorded the canonical Move into
      // moveHistory/history. Capture snapshots only so HistoryPlayback remains aligned.
      this.recordHistorySnapshotsOnly(beforeState);
    } else if (phase === 'movement' || phase === 'capture') {
      // Human-driven movement click. Record canonical history via the
      // movement engine hooks without interfering with canonical replays.
      this._movementInvocationContext = 'human';
      try {
        await this.handleMovementClick(pos);
      } finally {
        this._movementInvocationContext = null;
      }
    } else if (phase === 'chain_capture') {
      await this.handleChainCaptureClick(pos);
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
      getValidMovesForCurrentPlayer: () => this.getValidMoves(this.gameState.currentPlayer),
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
      canCurrentPlayerSwapSides: () => this.canCurrentPlayerSwapSides(),
      applySwapSidesForCurrentPlayer: () => this.applySwapSidesForCurrentPlayer(),
      getAIDifficulty: (playerNumber: number) => this.getAIDifficulty(playerNumber),
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

    // RR-FIX-2026-01-11: Respect mustMoveFromStackKey constraint.
    // If a constraint is active and `from` doesn't match, return empty.
    // This ensures UI highlights only show valid targets per the constraint.
    if (this._mustMoveFromStackKey && this._mustMoveFromStackKey !== fromKey) {
      return [];
    }

    // 1. Enumerate capture segments from this stack.
    const captureSegments = this.enumerateCaptureSegmentsFrom(from, playerNumber);
    const captureLandings = captureSegments.map((seg) => seg.landing);

    // 2. Enumerate simple (non-capturing) movement options from this stack
    // only during the core movement phase. In capture/chain_capture phases,
    // rules semantics allow only capture segments.
    //
    // EXCEPTION: In ring_placement, we also enumerate movement options to
    // support the "skip placement + move" interaction where selecting a stack
    // highlights its potential moves.
    let simpleLandings: Position[] = [];
    if (
      this.gameState.currentPhase === 'movement' ||
      this.gameState.currentPhase === 'ring_placement'
    ) {
      const simpleMoves = this.enumerateSimpleMovementLandings(playerNumber).filter(
        (m) => m.fromKey === fromKey
      );
      simpleLandings = simpleMoves.map((m) => m.to);
    }

    // 3. Enumerate recovery slide targets if player is eligible for recovery
    // (RR-CANON-R110–R115). Recovery slides originate from marker positions,
    // not stacks, so we check if the 'from' position matches any recovery target.
    let recoveryLandings: Position[] = [];
    if (
      this.gameState.currentPhase === 'movement' &&
      isPlayerEligibleForRecovery(this.gameState, playerNumber)
    ) {
      const recoveryTargets = enumerateRecoverySlideLandings(this.gameState, playerNumber);
      const matchingRecovery = recoveryTargets.filter((t) => positionToString(t.from) === fromKey);
      recoveryLandings = matchingRecovery.map((t) => t.to);
    }

    // 4. Return the union of capture, simple, and recovery landings, deduplicated.
    const allLandings: Position[] = [];
    const seen = new Set<string>();

    for (const pos of [...captureLandings, ...simpleLandings, ...recoveryLandings]) {
      const key = positionToString(pos);
      if (seen.has(key)) continue;
      seen.add(key);
      allLandings.push(pos);
    }

    return allLandings;
  }

  /**
   * Enumerate legal ring placement positions for the given player using the
   * canonical PlacementAggregate enumerator. This keeps sandbox placement
   * highlighting aligned with the shared engine and backend RuleEngine.
   */
  private enumerateLegalRingPlacements(playerNumber: number): Position[] {
    const state = this.gameState;
    const player = state.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.ringsInHand <= 0) {
      return [];
    }

    return enumeratePlacementPositions(state, playerNumber);
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
      (pos: Position) => this.isValidPositionLocal(pos)
    );
  }

  /**
   * R172 helper: true if the given player has any real action available
   * at the start of their turn (placement, non-capture movement, overtaking
   * capture). Recovery and forced elimination do not count as real actions.
   *
   * Delegates to the shared hasAnyRealAction helper with sandbox-specific
   * move enumerators.
   */
  private hasAnyRealActionForPlayer(playerNumber: number): boolean {
    const result = hasAnyRealAction(this.gameState, playerNumber, {
      hasPlacement: (pn) => this.enumerateLegalRingPlacements(pn).length > 0,
      hasMovement: (pn) => this.enumerateSimpleMovementLandings(pn).length > 0,
      hasCapture: (pn) => {
        const board = this.gameState.board;
        for (const stack of board.stacks.values()) {
          if (stack.controllingPlayer !== pn) continue;
          if (this.enumerateCaptureSegmentsFrom(stack.position, pn).length > 0) {
            return true;
          }
        }
        return false;
      },
    });

    debugLog(isSandboxLpsDebugEnabled(), '[SandboxLPS] hasAnyRealActionForPlayer', {
      playerNumber,
      result,
    });

    return result;
  }

  /**
   * Determine whether the current sandbox state should expose a swap_sides
   * meta-move (pie rule) for Player 2.
   *
   * Delegates to the shared shouldOfferSwapSides() helper from swapSidesHelpers.ts.
   */
  private shouldOfferSwapSidesMetaMove(): boolean {
    return shouldOfferSwapSides(this.gameState);
  }

  /**
   * Public helper used by SandboxGameHost/tests to determine whether the
   * current player may invoke the pie rule.
   */
  public canCurrentPlayerSwapSides(): boolean {
    return this.shouldOfferSwapSidesMetaMove();
  }

  /**
   * True if the player has any material left: any rings in play or in hand.
   * Delegates to the shared playerHasMaterial helper.
   */
  private playerHasMaterialLocal(playerNumber: number): boolean {
    return playerHasMaterial(this.gameState, playerNumber);
  }

  /**
   * Apply the pie-rule style colour/seat swap for the current sandbox game.
   * See shouldOfferSwapSidesMetaMove for gate conditions.
   *
   * Returns true when the swap was applied; false when the gate conditions
   * are not satisfied.
   */
  public applySwapSidesForCurrentPlayer(): boolean {
    if (!this.shouldOfferSwapSidesMetaMove()) {
      return false;
    }

    const state = this.gameState;
    const before = this.getGameState();

    const p1 = state.players.find((p) => p.playerNumber === 1);
    const p2 = state.players.find((p) => p.playerNumber === 2);

    if (!p1 || !p2) {
      return false;
    }

    const swappedPlayers = state.players.map((p) => {
      if (p.playerNumber === 1) {
        return {
          ...p,
          id: p2.id,
          username: p2.username,
          type: p2.type,
          rating: p2.rating,
          aiDifficulty: p2.aiDifficulty,
          aiProfile: p2.aiProfile,
        };
      }
      if (p.playerNumber === 2) {
        return {
          ...p,
          id: p1.id,
          username: p1.username,
          type: p1.type,
          rating: p1.rating,
          aiDifficulty: p1.aiDifficulty,
          aiProfile: p1.aiProfile,
        };
      }
      return p;
    }) as Player[];

    const moveNumber = state.moveHistory.length + 1;
    const move: Move = {
      id: `swap_sides-${moveNumber}`,
      type: 'swap_sides',
      player: state.currentPlayer,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move;

    this.gameState = {
      ...state,
      players: swappedPlayers,
      moveHistory: [...state.moveHistory, move],
      lastMoveAt: move.timestamp,
    };

    this.appendHistoryEntry(before, move);
    return true;
  }

  /**
   * Update last-player-standing round tracking for the current player.
   *
   * Delegates to the shared updateLpsTracking() helper from lpsTracking.ts.
   */
  private updateLpsRoundTrackingForCurrentPlayer(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    if (!isLpsActivePhase(this.gameState.currentPhase)) {
      return;
    }

    const state = this.gameState;
    const currentPlayer = state.currentPlayer;

    const activePlayers = state.players
      .filter((p) => this.playerHasMaterialLocal(p.playerNumber))
      .map((p) => p.playerNumber);

    const hasRealAction = this.hasAnyRealActionForPlayer(currentPlayer);

    // Delegate to shared helper which mutates _lpsState in place
    updateLpsTracking(this._lpsState, {
      currentPlayer,
      activePlayers,
      hasRealAction,
    });
  }

  // NOTE: finalizeCompletedLpsRound is now handled internally by updateLpsTracking()

  /**
   * Build a GameResult for a last-player-standing victory.
   * Delegates to the shared buildLpsVictoryResult() helper from lpsTracking.ts.
   */
  private buildLastPlayerStandingResult(winner: number): GameResult {
    return buildLpsVictoryResult(this.gameState, winner);
  }

  /**
   * Check whether R172 is satisfied at the start of the current player's
   * interactive turn and, if so, end the sandbox game with an LPS result.
   *
   * Delegates to the shared evaluateLpsVictory() helper from lpsTracking.ts.
   */
  private maybeEndGameByLastPlayerStanding(): void {
    const state = this.gameState;

    const lpsResult = evaluateLpsVictory({
      gameState: state,
      lps: this._lpsState,
      hasAnyRealAction: (pn) => this.hasAnyRealActionForPlayer(pn),
      hasMaterial: (pn) => this.playerHasMaterialLocal(pn),
    });

    debugLog(isSandboxLpsDebugEnabled(), '[SandboxLPS] evaluateLpsVictory', {
      stateSnapshot: {
        currentPlayer: state.currentPlayer,
        currentPhase: state.currentPhase,
        gameStatus: state.gameStatus,
        players: state.players.map((p) => ({
          playerNumber: p.playerNumber,
          ringsInHand: p.ringsInHand,
          eliminatedRings: p.eliminatedRings,
          territorySpaces: p.territorySpaces,
        })),
      },
      lpsState: {
        roundIndex: this._lpsState.roundIndex,
        currentRoundFirstPlayer: this._lpsState.currentRoundFirstPlayer,
        exclusivePlayerForCompletedRound: this._lpsState.exclusivePlayerForCompletedRound,
      },
      result: lpsResult,
    });

    if (!lpsResult.isVictory || lpsResult.winner === undefined) {
      return;
    }

    const winner = lpsResult.winner;
    const result = this.buildLastPlayerStandingResult(winner);

    this.gameState = {
      ...state,
      gameStatus: 'completed',
      winner,
      // Set terminal phase for semantic clarity and TS↔Python parity
      currentPhase: 'game_over',
    };
    this.victoryResult = result;

    // Reset LPS tracking state after a terminal outcome
    this._lpsState = createLpsTrackingState();
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
      lpsRoundIndex: this._lpsState.roundIndex,
      lpsCurrentRoundFirstPlayer: this._lpsState.currentRoundFirstPlayer,
      lpsExclusivePlayerForCompletedRound: this._lpsState.exclusivePlayerForCompletedRound,
    };

    this.updateLpsRoundTrackingForCurrentPlayer();
    this.maybeEndGameByLastPlayerStanding();

    const afterSnapshot = {
      currentPlayer: this.gameState.currentPlayer,
      currentPhase: this.gameState.currentPhase,
      gameStatus: this.gameState.gameStatus,
      lpsRoundIndex: this._lpsState.roundIndex,
      lpsCurrentRoundFirstPlayer: this._lpsState.currentRoundFirstPlayer,
      lpsExclusivePlayerForCompletedRound: this._lpsState.exclusivePlayerForCompletedRound,
    };

    debugLog(isSandboxLpsDebugEnabled(), '[TurnTrace.sandbox.handleStartOfInteractiveTurn]', {
      decision: 'handleStartOfInteractiveTurn',
      reason: 'start_interactive_turn',
      before: beforeSnapshot,
      after: afterSnapshot,
    });

    // RR-FIX-2026-01-12: Auto-advance for human players with 0 rings in ring_placement phase.
    // Per RR-CANON-R073, all players start in ring_placement, but players with 0 rings
    // must emit no_placement_action to advance to movement phase. AI handles this in
    // sandboxAI.ts, but human players need explicit handling here.
    this.maybeAutoAdvanceHumanWithNoRings();
  }

  /**
   * Auto-advance to movement phase for human players with 0 rings.
   * Called at the start of a turn to handle the case where a human player
   * has no rings in hand but needs to move from existing stacks.
   */
  private maybeAutoAdvanceHumanWithNoRings(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    // Only handle ring_placement phase
    if (this.gameState.currentPhase !== 'ring_placement') {
      return;
    }

    // Skip if we've already placed this turn (handled by handleHumanCellClick)
    if (this._mustMoveFromStackKey) {
      return;
    }

    const currentPlayer = this.gameState.players.find(
      (p) => p.playerNumber === this.gameState.currentPlayer
    );
    if (!currentPlayer) {
      return;
    }

    // Only handle human players (AI is handled by sandboxAI.ts)
    if (currentPlayer.type === 'ai') {
      return;
    }

    // Check if player has 0 rings in hand
    const ringsInHand = currentPlayer.ringsInHand ?? 0;
    if (ringsInHand > 0) {
      return;
    }

    // Human player with 0 rings in ring_placement phase - need to auto-advance.
    // Use setTimeout to defer so current execution can complete.
    const playerNumber = currentPlayer.playerNumber;
    window.setTimeout(() => {
      // Double-check state hasn't changed
      if (
        this.gameState.gameStatus !== 'active' ||
        this.gameState.currentPhase !== 'ring_placement' ||
        this.gameState.currentPlayer !== playerNumber
      ) {
        return;
      }

      const moveNumber = this.gameState.history.length + 1;
      const noPlacementMove: Move = {
        type: 'no_placement_action',
        player: playerNumber,
        id: `no-placement-action-human-0rings-${moveNumber}`,
        moveNumber,
        timestamp: new Date(),
        thinkTime: 0,
        to: { x: 0, y: 0 },
      } as Move;

      this.applyCanonicalMove(noPlacementMove).catch((err) => {
        console.error('[ClientSandboxEngine] Failed to auto-advance human with 0 rings:', err);
      });
    }, 50);
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
      isValidPosition: (pos) => this.isValidPositionLocal(pos),
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
    // Use the shared enumerateAllCaptureMovesAggregate helper but scoped to a single position.
    // This ensures we use the exact same logic as the backend and shared engine.
    // We construct a temporary GameState that wraps the current board so the aggregate
    // can inspect it.
    const tempState: GameState = {
      ...this.gameState,
      board,
      currentPlayer: playerNumber,
    };

    // The aggregate enumerates ALL captures for the player. We filter for the specific 'from' position.
    const allCaptures = enumerateAllCaptureMovesAggregate(tempState, playerNumber);
    const fromKey = positionToString(from);

    return allCaptures
      .filter((m) => m.from && positionToString(m.from) === fromKey)
      .map((m) => ({
        from: m.from as Position,
        target: m.captureTarget as Position,
        landing: m.to as Position,
      }));
  }

  /**
   * Replay-only helper: detect whether the current player has any
   * overtaking capture segments available from any of their stacks in
   * the current snapshot. Used by the self-play replay harness to
   * distinguish between a true, ongoing capture phase (where explicit
   * continuation moves should follow) and a fully-resolved capture
   * where the backend has already advanced the turn.
   */
  private hasAnyCaptureSegmentsForCurrentPlayer(): boolean {
    const state = this.gameState;
    const playerNumber = state.currentPlayer;
    const stacks = this.getPlayerStacks(playerNumber, state.board);

    for (const stack of stacks) {
      const segments = this.enumerateCaptureSegmentsFrom(stack.position, playerNumber);
      if (segments.length > 0) {
        return true;
      }
    }

    return false;
  }

  private applyCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): void {
    const beforeState = this.gameState;

    const outcome = applyCaptureSegmentAggregate(beforeState, {
      from,
      target,
      landing,
      player: playerNumber,
    });

    this.gameState = outcome.nextState;
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
    await this.performCaptureChainInternal(from, target, landing, playerNumber);
  }

  private createTurnLogicDelegates(): TurnLogicDelegates {
    return {
      getPlayerStacks: (state, playerNumber) => this.getPlayerStacks(playerNumber, state.board),
      // Use shared canonical predicates from turnDelegateHelpers.ts
      hasAnyPlacement: (state, playerNumber) => hasAnyPlacementForPlayer(state, playerNumber),
      hasAnyMovement: (state, playerNumber, turn) =>
        hasAnyMovementForPlayer(state, playerNumber, turn as SandboxTurnState),
      hasAnyCapture: (state, playerNumber, turn) =>
        hasAnyCaptureForPlayer(state, playerNumber, turn as SandboxTurnState),
      applyForcedElimination: (state, playerNumber) => {
        // In traceMode, do not inject host-level forced elimination between
        // recorded moves. Canonical replays expect any eliminations to be
        // represented as explicit moves in the history, so the delegate is
        // a no-op here. Live games (traceMode=false) keep the original
        // behaviour.
        if (this.traceMode) {
          return state;
        }

        this.gameState = state;
        // Use sync version for delegate (TurnLogicDelegates requires sync execution)
        this.forceEliminateCapSync(playerNumber);
        this.checkAndApplyVictory();
        return this.gameState;
      },
      getNextPlayerNumber: (state, current) => this.getNextPlayerNumber(current),
      playerHasAnyRings: (state, player) => playerHasAnyRings(state, player),
    };
  }

  // hasAnyMovementOrCaptureForPlayer removed - now uses shared predicates:
  // - hasAnyMovementForPlayer from turnDelegateHelpers.ts
  // - hasAnyCaptureForPlayer from turnDelegateHelpers.ts

  private startTurnForCurrentPlayer(): void {
    // Before starting a new turn, re-check victory in case prior
    // movement/territory/line processing produced a terminal state.
    this.checkAndApplyVictory();
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    let turnState: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    // Reset per-turn flags at the beginning of a player's turn.
    turnState = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

    // We may need to advance through multiple players if some are forced
    // to eliminate a cap and immediately lose their turn.
    for (let safety = 0; safety < this.gameState.players.length; safety++) {
      const current = this.gameState.currentPlayer;
      const player = this.gameState.players.find((p) => p.playerNumber === current);
      if (!player) {
        return;
      }

      const eliminatedResult = this.maybeProcessForcedEliminationForCurrentPlayerInternal(
        this.gameState,
        turnState
      );
      this.gameState = eliminatedResult.state;
      turnState = eliminatedResult.turnState;

      if (eliminatedResult.eliminated) {
        // Continue loop with updated current player.
        continue;
      }

      // Per RR-CANON-R073: ALL players start in ring_placement without exception.
      // NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action.
      this.gameState = {
        ...this.gameState,
        currentPhase: 'ring_placement', // Always ring_placement - NO PHASE SKIPPING
      };

      this._hasPlacedThisTurn = turnState.hasPlacedThisTurn;
      this._mustMoveFromStackKey = turnState.mustMoveFromStackKey;

      this.handleStartOfInteractiveTurn();
      return;
    }
  }

  private maybeProcessForcedEliminationForCurrentPlayer(): boolean {
    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const result = this.maybeProcessForcedEliminationForCurrentPlayerInternal(
      this.gameState,
      turnStateBefore
    );

    this.gameState = result.state;
    this._hasPlacedThisTurn = result.turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = result.turnState.mustMoveFromStackKey;

    return result.eliminated;
  }

  private maybeProcessForcedEliminationForCurrentPlayerInternal(
    state: GameState,
    turnState: SandboxTurnState
  ): { state: GameState; turnState: SandboxTurnState; eliminated: boolean } {
    // In strict replay/trace mode we must not inject implicit forced
    // eliminations between recorded moves. Canonical self-play replays
    // represent such effects as explicit eliminate_rings_from_stack moves
    // in the history. When traceMode is enabled, treat this helper as a
    // no-op so that replay semantics are driven solely by the recorded
    // move sequence.
    if (this.traceMode) {
      return { state, turnState, eliminated: false };
    }

    const current = state.currentPlayer;
    const player = state.players.find((p) => p.playerNumber === current);
    if (!player) {
      return { state, turnState, eliminated: false };
    }

    const board = state.board;
    const stacks = this.getPlayerStacks(current, board);
    if (stacks.length === 0) {
      if (player.ringsInHand <= 0) {
        const nextPlayer = this.getNextPlayerNumber(current);
        const nextState: GameState = {
          ...state,
          currentPlayer: nextPlayer,
        };

        const nextTurnState: SandboxTurnState = {
          hasPlacedThisTurn: false,
          mustMoveFromStackKey: undefined,
        };

        return { state: nextState, turnState: nextTurnState, eliminated: true };
      }

      return { state, turnState, eliminated: false };
    }

    const mustKey = turnState.mustMoveFromStackKey;
    let hasAnyAction: boolean;
    const nextTurnState = { ...turnState };

    if (mustKey && state.currentPhase === 'movement') {
      const mustStack = stacks.find((s) => positionToString(s.position) === mustKey);

      if (mustStack) {
        hasAnyAction = this.hasAnyLegalMoveOrCaptureFrom(mustStack.position, current, board);
      } else {
        nextTurnState.mustMoveFromStackKey = undefined;
        hasAnyAction = stacks.some((stack) =>
          this.hasAnyLegalMoveOrCaptureFrom(stack.position, current, board)
        );
      }
    } else {
      hasAnyAction = stacks.some((stack) =>
        this.hasAnyLegalMoveOrCaptureFrom(stack.position, current, board)
      );
    }

    const hasAnyPlacement = (() => {
      if (player.ringsInHand <= 0) {
        return false;
      }
      const placements = enumeratePlacementPositions(state, current);
      return placements.length > 0;
    })();

    if (hasAnyAction || hasAnyPlacement) {
      return { state, turnState: nextTurnState, eliminated: false };
    }

    // Forced elimination: eliminate a cap and advance to next player.
    // Use sync version here as this is a synchronous turn-transition path.
    this.gameState = state;
    this.forceEliminateCapSync(current);
    let nextState = this.gameState;

    const nextPlayer = this.getNextPlayerNumber(current);
    nextState = {
      ...nextState,
      currentPlayer: nextPlayer,
    };

    const resetTurnState: SandboxTurnState = {
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    };

    return { state: nextState, turnState: resetTurnState, eliminated: true };
  }

  private advanceTurnAndPhaseForCurrentPlayer(): void {
    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const delegates = this.createTurnLogicDelegates();

    const { nextState, nextTurn } = advanceTurnAndPhase(this.gameState, turnStateBefore, delegates);

    this.gameState = nextState;
    this._hasPlacedThisTurn = nextTurn.hasPlacedThisTurn;
    this._mustMoveFromStackKey = nextTurn.mustMoveFromStackKey;

    this.handleStartOfInteractiveTurn();
  }

  /**
   * Sync forced elimination helper for blocked players - uses
   * applyForcedEliminationForPlayer which checks preconditions (player must
   * be blocked with no legal actions). Used by TurnLogicDelegates.
   */
  private forceEliminateCapSync(playerNumber: number): void {
    // In traceMode (canonical replay), forced elimination must only occur
    // via explicit recorded moves (normalized as eliminate_rings_from_stack).
    // Suppress the implicit helper so that replays do not introduce extra
    // eliminations between moves.
    if (this.traceMode) {
      return;
    }

    const outcome = applyForcedEliminationForPlayer(this.gameState, playerNumber);
    if (!outcome || outcome.eliminatedCount <= 0) {
      return;
    }
    this.gameState = outcome.nextState;
  }

  /**
   * Unconditional ring elimination for line rewards - eliminates ONE ring from the
   * player's stacks without checking forced elimination preconditions. Used
   * for exact-length lines and collapse-all line reward options.
   *
   * Per RR-CANON-R122: Line processing requires eliminating exactly ONE ring
   * from any controlled stack (including standalone rings).
   */
  private eliminateRingForLineReward(playerNumber: number): void {
    const { board, players } = this.gameState;
    const stacks = this.getPlayerStacks(playerNumber, board);

    // Pass 'line' context to eliminate only ONE ring per RR-CANON-R122
    const result = forceEliminateCapOnBoard(board, players, playerNumber, stacks, 'line');
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

  /**
   * Async forced elimination helper - presents a RingEliminationChoice to
   * human players when multiple stacks are available. For AI players or
   * single-stack scenarios, auto-selects immediately.
   */
  private async forceEliminateCap(playerNumber: number): Promise<void> {
    const { players } = this.gameState;

    // Use the enumerate function to get all elimination options
    const options = enumerateForcedEliminationOptions(this.gameState, playerNumber);
    if (options.length === 0) {
      return; // No elimination needed
    }

    // Determine target position - either from player choice or auto-select
    let targetPosition: Position | undefined;

    // Check if this is a human player with multiple stacks (should present choice)
    const player = players.find((p) => p.playerNumber === playerNumber);
    const isHumanPlayer = player?.type === 'human';
    const hasMultipleOptions = options.length > 1;

    if (isHumanPlayer && hasMultipleOptions) {
      // Present RingEliminationChoice to human player
      const choice: RingEliminationChoice = {
        id: `sandbox-forced-elim-${Date.now()}`,
        gameId: this.gameState.id,
        playerNumber,
        type: 'ring_elimination',
        eliminationContext: 'forced',
        prompt: 'Forced elimination: You must eliminate your ENTIRE CAP from a controlled stack.',
        options: options.map((opt) => ({
          stackPosition: opt.position,
          capHeight: opt.capHeight,
          totalHeight: opt.stackHeight,
          // Forced elimination removes entire cap (RR-CANON-R100)
          ringsToEliminate: opt.capHeight,
          moveId: opt.moveId,
        })),
      };

      const response = await this.interactionHandler.requestChoice(choice);
      targetPosition = response.selectedOption.stackPosition;
    } else if (options.length === 1) {
      // Only one option, use it directly
      targetPosition = options[0].position;
    }
    // If no targetPosition, applyForcedEliminationForPlayer will auto-select

    // Apply elimination using the shared helper with the chosen target
    const outcome = applyForcedEliminationForPlayer(this.gameState, playerNumber, targetPosition);
    if (!outcome || outcome.eliminatedCount <= 0) {
      return;
    }

    this.gameState = outcome.nextState;
  }

  private getNextPlayerNumber(current: number): number {
    const players = this.gameState.players;
    const idx = players.findIndex((p) => p.playerNumber === current);
    const nextIdx = (idx + 1) % players.length;
    return players[nextIdx].playerNumber;
  }

  /**
   * Local position validity check delegating to the shared helper
   * for consistent semantics with the backend BoardManager.
   */
  private isValidPositionLocal(pos: Position): boolean {
    const config = BOARD_CONFIGS[this.gameState.boardType];
    return isValidPosition(pos, this.gameState.boardType, config.size);
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
    const wasCollapsed = board.collapsedSpaces.has(key);

    // When a marker collapses to territory, the cell becomes
    // exclusive territory: no stacks or markers may remain.
    board.markers.delete(key);
    board.stacks.delete(key);
    board.collapsedSpaces.set(key, playerNumber);

    // Mirror shared-engine semantics: any newly-collapsed space created
    // via movement or capture should increment the owning player's
    // territorySpaces exactly once.
    if (!wasCollapsed) {
      const updatedPlayers = this.gameState.players.map((p) =>
        p.playerNumber === playerNumber
          ? {
              ...p,
              territorySpaces: (p.territorySpaces ?? 0) + 1,
            }
          : p
      );

      this.gameState = {
        ...this.gameState,
        players: updatedPlayers,
      };
    }
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
      (pos: Position) => this.isValidPositionLocal(pos),
      stringToPosition
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

    // Use the extracted choice builder for capture direction prompts
    const choice = buildCaptureDirectionChoice(
      this.gameState.id,
      this.gameState.currentPlayer,
      captureOptions,
      this.gameState.board
    );

    if (choice.options.length === 0) {
      return captureOptions[0];
    }

    const response = await this.interactionHandler.requestChoice(choice);

    // Find the matching move based on response
    const typedCaptures = captureOptions.filter(isCaptureMove);
    const selected = typedCaptures.find((move) => {
      const opt = response.selectedOption;
      return (
        opt &&
        positionToString(opt.targetPosition) === positionToString(move.captureTarget) &&
        positionToString(opt.landingPosition) === positionToString(move.to)
      );
    });

    return selected ?? captureOptions[0];
  }

  /**
   * Handle a human click during the chain_capture phase by mapping the clicked
   * landing position onto a canonical continue_capture_segment Move from the
   * orchestrator adapter and applying it via applyCanonicalMove.
   */
  private async handleChainCaptureClick(position: Position): Promise<void> {
    const state = this.getGameState();

    if (state.gameStatus !== 'active' || state.currentPhase !== 'chain_capture') {
      return;
    }

    const adapter = this.getOrchestratorAdapter();
    const allMoves = adapter.getValidMoves();
    if (!allMoves || allMoves.length === 0) {
      return;
    }

    const currentPlayer = state.currentPlayer;
    const targetKey = positionToString(position);

    const matching = allMoves.filter((m) => {
      if (m.player !== currentPlayer) return false;
      if (m.type !== 'continue_capture_segment') return false;
      if (!m.to) return false;
      return positionToString(m.to as Position) === targetKey;
    });

    if (matching.length === 0) {
      // Click was not on a valid continuation landing; ignore.
      return;
    }

    const chosen = matching[0];

    await this.applyCanonicalMove(chosen);
  }

  private async handleMovementClick(position: Position): Promise<void> {
    const board = this.gameState.board;
    const key = positionToString(position);
    const stackAtPos = board.stacks.get(key);

    // Synchronous selection / deselection logic to preserve existing
    // click-to-select semantics used by tests and the UI.
    if (!this.isValidPositionLocal(position)) {
      this._selectedStackKey = undefined;
      return;
    }

    if (!this._selectedStackKey) {
      // In capture phase, allow clicking directly on a highlighted landing
      // cell to apply an overtaking_capture without pre-selecting the stack.
      if (this.gameState.currentPhase === 'capture') {
        const adapter = this.getOrchestratorAdapter();
        const validMoves = adapter.getValidMoves();
        const captureMoves = validMoves.filter(
          (m) =>
            m.type === 'overtaking_capture' && m.to && positionToString(m.to as Position) === key
        );

        if (captureMoves.length > 0) {
          let moveToApply: Move = captureMoves[0];

          // If multiple capture options share the same landing, delegate to the
          // existing capture-direction choice helper to disambiguate.
          if (captureMoves.length > 1) {
            moveToApply = await this.promptForCaptureDirection(captureMoves);
          }

          await this.applyCanonicalMove(moveToApply);
          return;
        }
      }

      // Otherwise fall back to normal click-to-select semantics.
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
    // Adapter-based movement (orchestrator is permanently enabled)
    // ═══════════════════════════════════════════════════════════════════════
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
  }

  private async performCaptureChainInternal(
    initialFrom: Position,
    initialTarget: Position,
    initialLanding: Position,
    playerNumber: number,
    isCanonicalReplay: boolean = false
  ): Promise<void> {
    let state = this.gameState;
    let currentPosition = initialLanding;
    let from = initialFrom;
    let target = initialTarget;
    let landing = initialLanding;
    let segmentIndex = 0;

    let before = this.gameState;
    this.applyCaptureSegment(from, target, landing, playerNumber);

    state = this.gameState;
    if (state.currentPhase !== 'chain_capture') {
      this.gameState = {
        ...state,
        currentPhase: 'chain_capture',
      };
    }

    let after = this.gameState;

    let pendingSegment: {
      before: GameState;
      after: GameState;
      from: Position;
      target: Position;
      landing: Position;
      playerNumber: number;
      segmentIndex: number;
    } | null = {
      before,
      after,
      from,
      target,
      landing,
      playerNumber,
      segmentIndex,
    };

    while (true) {
      state = this.gameState;
      const options = this.enumerateCaptureSegmentsFrom(currentPosition, playerNumber);

      if (options.length === 0) {
        await this.advanceAfterMovement();

        if (!isCanonicalReplay && pendingSegment) {
          await this.handleCaptureSegmentApplied({
            ...pendingSegment,
            after: this.gameState,
            isFinal: true,
          });
        }

        return;
      }

      if (!isCanonicalReplay && pendingSegment) {
        await this.handleCaptureSegmentApplied({
          ...pendingSegment,
          isFinal: false,
        });
      }

      let nextSegment: { from: Position; target: Position; landing: Position } | undefined;

      if (options.length > 1) {
        const captureMoves = options.map((opt) => ({
          from: opt.from,
          targetPosition: opt.target,
          landingPosition: opt.landing,
        }));

        const choice: CaptureDirectionChoice = {
          id: `sandbox-capture-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          gameId: state.id,
          playerNumber,
          type: 'capture_direction',
          prompt: 'Select capture direction',
          options: captureMoves.map((opt) => ({
            targetPosition: opt.targetPosition,
            landingPosition: opt.landingPosition,
            capturedCapHeight:
              state.board.stacks.get(positionToString(opt.targetPosition))?.capHeight ?? 0,
          })),
        };

        const response = await this.interactionHandler.requestChoice(choice);
        const selected = options.find((opt) => {
          const o = response.selectedOption;
          return (
            o &&
            positionToString(o.targetPosition) === positionToString(opt.target) &&
            positionToString(o.landingPosition) === positionToString(opt.landing)
          );
        });

        nextSegment = selected ?? options[0];
      } else {
        nextSegment = options[0];
      }

      if (!nextSegment) {
        return;
      }

      from = currentPosition;
      target = nextSegment.target;
      landing = nextSegment.landing;

      before = this.gameState;
      this.applyCaptureSegment(from, target, landing, playerNumber);
      after = this.gameState;

      segmentIndex += 1;
      pendingSegment = {
        before,
        after,
        from,
        target,
        landing,
        playerNumber,
        segmentIndex,
      };

      currentPosition = landing;
    }
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
      debugLog(
        SANDBOX_AI_STALL_DIAGNOSTICS,
        '[ClientSandboxEngine.advanceAfterMovement] EARLY RETURN: line_processing'
      );
      return;
    }

    await this.processDisconnectedRegionsForCurrentPlayer();
    this.debugCheckpoint('after-processDisconnectedRegionsForCurrentPlayer');

    // Same traceMode handling for territory_processing
    if (this.gameState.currentPhase === 'territory_processing') {
      debugLog(
        SANDBOX_AI_STALL_DIAGNOSTICS,
        '[ClientSandboxEngine.advanceAfterMovement] EARLY RETURN: territory_processing',
        { currentPlayer: this.gameState.currentPlayer }
      );
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

    debugLog(
      SANDBOX_AI_STALL_DIAGNOSTICS,
      '[ClientSandboxEngine.advanceAfterMovement] Calling advanceTurnAndPhaseForCurrentPlayer',
      { currentPlayer: this.gameState.currentPlayer }
    );

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

    let state = this.gameState;
    const movingPlayer = state.currentPlayer;
    let pendingSelfElimination = false;

    // Keep processing until no further eligible regions remain.
    while (true) {
      const disconnected = findDisconnectedRegionsOnBoard(state.board);
      if (disconnected.length === 0) {
        break;
      }

      const eligible: Territory[] = disconnected.filter((region) =>
        this.canProcessDisconnectedRegion(region.spaces, movingPlayer, state.board)
      );

      if (eligible.length === 0) {
        break;
      }

      let regionSpaces = eligible[0].spaces;

      const movingPlayerRecord = state.players.find((p) => p.playerNumber === movingPlayer);
      const isHumanPlayer = movingPlayerRecord?.type === 'human';

      // When at least one region is eligible for a human player and we have
      // an interaction handler, surface a RegionOrderChoice so the sandbox UI
      // can highlight the region geometry and prompt the user. When only one
      // region exists this still produces a single-option choice, mirroring
      // backend decision-phase semantics while remaining trivial to resolve.
      const shouldPromptRegionOrder =
        !!this.interactionHandler && isHumanPlayer && !this.traceMode && eligible.length > 0;

      if (shouldPromptRegionOrder) {
        // Populate board.territories with the eligible regions so that:
        // 1. gameViewModels.ts can highlight all cells in each region
        // 2. useSandboxInteractions.ts click handler can match clicks to regions
        // Keys must match the regionId values used in the choice options.
        // Use `region-${index}` format for consistency with sandboxDecisionMapping.ts
        const newTerritories = new Map(state.board.territories);
        eligible.forEach((region, index) => {
          newTerritories.set(`region-${index}`, region);
        });
        state = {
          ...state,
          board: {
            ...state.board,
            territories: newTerritories,
          },
        };
        // Also update the instance state so getGameState() returns the populated territories
        this.gameState = state;

        const choice: RegionOrderChoice = {
          id: `sandbox-region-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          gameId: state.id,
          playerNumber: movingPlayer,
          type: 'region_order',
          prompt: 'Territory claimed – choose area to process',
          options: eligible.map((r, index) => {
            const representative = r.spaces[0];
            // RR-FIX-2026-01-13: Use geometry-based stable regionId instead of array index.
            // Array indices change after each region is processed, causing wrong region selection.
            const stableRegionId = representative
              ? `region-${representative.x}-${representative.y}-${r.spaces.length}`
              : `region-fallback-${index}`;
            const regionKey = representative
              ? `${representative.x},${representative.y}`
              : `region-${index}`;

            return {
              regionId: stableRegionId,
              size: r.spaces.length,
              representativePosition: representative,
              moveId: `process-region-${stableRegionId}-${regionKey}`,
              // RR-FIX-2026-01-12: Include full region geometry for highlighting.
              // This ensures successive territories are highlighted correctly
              // since deriveBoardDecisionHighlights uses these positions directly
              // instead of looking up from potentially stale gameState.board.territories.
              spaces: r.spaces,
            };
          }),
        };

        const response = await this.interactionHandler.requestChoice(choice);
        const selected = response.selectedOption;
        // RR-FIX-2026-01-13: Match by moveId for stable region identification instead of
        // extracting index from regionId (which is now geometry-based, not index-based).
        const matchingOption = choice.options.find((opt) => opt.moveId === selected.moveId);
        const optionIndex = matchingOption ? choice.options.indexOf(matchingOption) : 0;
        const selectedRegion = eligible[optionIndex] ?? eligible[0];
        regionSpaces = selectedRegion.spaces;
      }

      const beforeSnapshot = {
        collapsedSpaces: state.board.collapsedSpaces.size,
        totalRingsEliminated: state.totalRingsEliminated,
      };

      // Apply region-processing consequences via the shared helper using a
      // synthetic choose_territory_option Move so automatic processing
      // shares semantics with explicit move application.
      const regionMove: Move = {
        id: `auto-process-region-${Date.now()}`,
        type: 'choose_territory_option',
        player: movingPlayer,
        disconnectedRegions: [
          {
            spaces: regionSpaces,
            controllingPlayer: movingPlayer,
            isDisconnected: true,
          },
        ],
        to: regionSpaces[0] ?? { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: state.history.length + 1,
      } as Move;

      const outcome = applyProcessTerritoryRegionDecision(state, regionMove);
      state = outcome.nextState;

      // Sanity check: compare collapsed spaces before/after to ensure actual
      // territory was claimed. This guards against edge cases where region
      // detection finds a region but processing results in no state change.
      const collapsedAfterProcessing = state.board.collapsedSpaces.size;
      const actualTerritoryGain = collapsedAfterProcessing - beforeSnapshot.collapsedSpaces;

      // If this region triggered a self-elimination requirement, either
      // surface an explicit elimination choice for human players or fall
      // back to automatic elimination for AI/legacy flows.
      // Skip elimination if no actual territory was gained (defensive guard).
      if (outcome.pendingSelfElimination && actualTerritoryGain > 0) {
        pendingSelfElimination = true;

        const territoryEliminationContext = ((): 'territory' | 'recovery' => {
          // Recovery-context territory self-elimination (RR-CANON-R114) applies
          // when the current turn included a recovery_slide.
          for (let i = state.moveHistory.length - 1; i >= 0; i--) {
            const prior = state.moveHistory[i];
            if (prior.player !== movingPlayer) {
              break;
            }
            if (prior.type === 'recovery_slide') {
              return 'recovery';
            }
          }
          return 'territory';
        })();

        const shouldPromptElimination =
          !!this.interactionHandler && isHumanPlayer && !this.traceMode;

        const processedRegion: Territory = {
          spaces: regionSpaces,
          controllingPlayer: movingPlayer,
          isDisconnected: true,
        };

        const eliminationMoves = enumerateTerritoryEliminationMoves(state, movingPlayer, {
          eliminationContext: territoryEliminationContext,
          processedRegion,
        });

        if (shouldPromptElimination) {
          // Mark the pending flag so parity/diagnostic helpers that inspect
          // _pendingTerritorySelfElimination remain meaningful while an
          // explicit ring_elimination decision is outstanding.
          this._pendingTerritorySelfElimination = true;

          if (eliminationMoves.length > 0) {
            let elimMove: Move = eliminationMoves[0];

            // RR-FIX-2026-01-10: Always show elimination UI for human players, even with a single option.
            // Previously only showed UI for multiple options (> 1), causing single-option eliminations
            // to be silently auto-applied without highlighting or visual feedback.
            if (eliminationMoves.length >= 1) {
              // Build a descriptive prompt explaining which spaces were claimed
              const spacesList = regionSpaces
                .slice(0, 4)
                .map((p) => `(${p.x},${p.y})`)
                .join(', ');
              const truncated = regionSpaces.length > 4 ? ` +${regionSpaces.length - 4} more` : '';
              const territoryPrompt =
                territoryEliminationContext === 'recovery'
                  ? `Territory claimed at ${spacesList}${truncated}. You must extract ONE buried ring from a stack outside the region.`
                  : `Territory claimed at ${spacesList}${truncated}. You must eliminate your ENTIRE CAP from an eligible stack outside the region.`;

              const choice: RingEliminationChoice = {
                id: `sandbox-territory-elim-${Date.now()}`,
                gameId: state.id,
                playerNumber: movingPlayer,
                type: 'ring_elimination',
                eliminationContext: territoryEliminationContext,
                prompt: territoryPrompt,
                options: eliminationMoves.map((opt: Move) => {
                  const pos = opt.to as Position;
                  const key = positionToString(pos);
                  const stack = state.board.stacks.get(key);

                  const capHeight =
                    (opt.eliminationFromStack && opt.eliminationFromStack.capHeight) ||
                    (stack ? stack.capHeight : 1);
                  const totalHeight =
                    (opt.eliminationFromStack && opt.eliminationFromStack.totalHeight) ||
                    (stack ? stack.stackHeight : capHeight || 1);

                  const ringsToEliminate =
                    typeof opt.eliminatedRings?.[0]?.count === 'number'
                      ? opt.eliminatedRings[0].count
                      : capHeight;

                  return {
                    stackPosition: pos,
                    capHeight,
                    totalHeight,
                    ringsToEliminate,
                    moveId: opt.id || key,
                  };
                }),
              };

              // RR-DEBUG-2026-01-10: Log territory elimination choice for debugging
              // eslint-disable-next-line no-console
              console.log('[ClientSandboxEngine] Building territory ring_elimination choice:', {
                choiceId: choice.id,
                playerNumber: choice.playerNumber,
                eliminationContext: territoryEliminationContext,
                optionsCount: choice.options.length,
                options: choice.options.map((opt) => ({
                  stackPosition: opt.stackPosition,
                  capHeight: opt.capHeight,
                  totalHeight: opt.totalHeight,
                  ringsToEliminate: opt.ringsToEliminate,
                })),
              });

              const response = await this.interactionHandler.requestChoice(choice);
              const selectedPos = response.selectedOption?.stackPosition as Position | undefined;

              if (selectedPos) {
                const selectedKey = positionToString(selectedPos);
                const byPos =
                  eliminationMoves.find(
                    (m) => m.to && positionToString(m.to as Position) === selectedKey
                  ) ?? eliminationMoves[0];
                elimMove = byPos;
              }
            }

            const beforeElim = state;
            const { nextState } = applyEliminateRingsFromStackDecision(state, elimMove);
            state = nextState;
            this.appendHistoryEntry(beforeElim, elimMove);
          }

          this._pendingTerritorySelfElimination = false;
        } else {
          if (eliminationMoves.length > 0) {
            const elimMove = eliminationMoves[0];
            const beforeElim = state;
            const { nextState } = applyEliminateRingsFromStackDecision(state, elimMove);
            state = nextState;
            this.appendHistoryEntry(beforeElim, elimMove);
          }
        }
      }

      const afterSnapshot = {
        collapsedSpaces: state.board.collapsedSpaces.size,
        totalRingsEliminated: state.totalRingsEliminated,
      };

      // Territory engine invariants: collapsedSpaces and totalRingsEliminated
      // should be monotonic across automatic territory processing.
      const errors: string[] = [];
      if (afterSnapshot.collapsedSpaces < beforeSnapshot.collapsedSpaces) {
        errors.push(
          `collapsedSpaces decreased in territory processing: before=${beforeSnapshot.collapsedSpaces}, after=${afterSnapshot.collapsedSpaces}`
        );
      }
      if (afterSnapshot.totalRingsEliminated < beforeSnapshot.totalRingsEliminated) {
        errors.push(
          `totalRingsEliminated decreased in territory processing: before=${beforeSnapshot.totalRingsEliminated}, after=${afterSnapshot.totalRingsEliminated}`
        );
      }
      if (errors.length > 0) {
        const message =
          'ClientSandboxEngine territory invariant violation (processDisconnectedRegionsForCurrentPlayer):\n' +
          errors.join('\n');

        console.error(message);
        if (isTestEnvironment()) {
          throw new Error(message);
        }
      }
    }

    this.gameState = state;

    // The in-loop self-elimination mirrors backend behavior; we only surface
    // the pending flag for traceMode callers that want explicit elimination
    // decision Moves instead of automatic application.
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
   * - choose_territory_option (legacy alias: process_territory_region): choose which eligible disconnected
   *   region to process first, subject to the self-elimination
   *   prerequisite from §12.2 / FAQ Q23.
   *
   * For normal human-driven sandbox games, territory collapse continues
   * to be driven automatically from advanceAfterMovement; this helper is
   * intended for canonical replay and advanced parity harnesses.
   */
  private getValidTerritoryProcessingMovesForCurrentPlayer(): Move[] {
    const state = this.gameState;
    const currentPlayer = state.currentPlayer;

    const rawMoves = enumerateProcessTerritoryRegionMoves(state, currentPlayer);
    if (rawMoves.length === 0) {
      return rawMoves;
    }

    return rawMoves.filter((move) => {
      if (!move.disconnectedRegions || move.disconnectedRegions.length === 0) {
        return false;
      }
      const region = move.disconnectedRegions[0];
      return this.canProcessDisconnectedRegion(region.spaces, currentPlayer, state.board);
    });
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

    // Per RR-CANON-R122 vs R145: Line elimination uses different rules
    // - Line: any controlled stack (including height-1), eliminate 1 ring
    // - Territory: any controlled stack (including height-1), eliminate entire cap
    const eliminationContext = pendingLineReward
      ? 'line'
      : (() => {
          for (let i = this.gameState.moveHistory.length - 1; i >= 0; i--) {
            const move = this.gameState.moveHistory[i];
            if (move.player !== movingPlayer) {
              break;
            }
            if (move.type === 'recovery_slide') {
              return 'recovery';
            }
          }
          return 'territory';
        })();
    return enumerateTerritoryEliminationMoves(this.gameState, movingPlayer, { eliminationContext });
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
    if (result) {
      debugLog(SANDBOX_AI_STALL_DIAGNOSTICS, '[ClientSandboxEngine Victory Debug]', {
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
   * built directly on top of the shared {@link enumerateProcessLineMoves}
   * and {@link enumerateChooseLineRewardMoves} helpers in
   * src/shared/engine/lineDecisionHelpers.ts, with geometry provided by
   * {@link findLinesForPlayer} from src/shared/engine/lineDetection.ts.
   */
  private getValidLineProcessingMovesForCurrentPlayer(): Move[] {
    const state = this.gameState;
    const currentPlayer = state.currentPlayer;

    const processMoves = enumerateProcessLineMoves(state, currentPlayer, {
      detectionMode: 'detect_now',
    });

    const playerLines = findLinesForPlayer(state.board, currentPlayer);
    const rewardMoves: Move[] = [];

    playerLines.forEach((_line, index) => {
      rewardMoves.push(...enumerateChooseLineRewardMoves(state, currentPlayer, index));
    });

    return [...processMoves, ...rewardMoves];
  }

  private async processLinesForCurrentPlayer(): Promise<void> {
    // Keep applying lines for the current player until none remain.
    // We use the shared lineDecisionHelpers (via
    // getValidLineProcessingMovesForCurrentPlayer) to identify
    // candidates and apply the corresponding canonical line-decision
    // helpers to execute them.
    //
    // Automatic sandbox behaviour remains:
    // - Exact-length lines: collapse all markers and immediately eliminate
    //   a cap via eliminateRingForLineReward.
    // - Overlength lines: default to Option 2 (minimum contiguous subset of
    //   length L, no elimination) for the first available line.
    //
    // While processing, accumulate the positions of any markers that collapse
    // to territory so the sandbox host can render a brief visual cue for the
    // newly-formed line segments.
    const recentLineKeys = new Set<string>();

    while (true) {
      const moves = this.getValidLineProcessingMovesForCurrentPlayer();
      const processLineMoves = moves.filter((m) => m.type === 'process_line');

      if (processLineMoves.length === 0) {
        break;
      }

      const boardType = this.gameState.boardType;
      const requiredLength = getEffectiveLineLengthThreshold(
        boardType,
        this.gameState.players.length,
        this.gameState.rulesOptions
      );

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
        // Use sorted position key for order-independent matching
        const lineKey = line.positions
          .map((p) => positionToString(p))
          .sort()
          .join('|');
        const rewardCandidates = moves.filter(
          (m) =>
            normalizeMoveType(m.type) === 'choose_line_option' &&
            m.formedLines &&
            m.formedLines.length > 0 &&
            m.formedLines[0].positions.length === line.positions.length &&
            m.formedLines[0].positions
              .map((p) => positionToString(p))
              .sort()
              .join('|') === lineKey
        );

        // RR-FIX-2026-01-12: For human players, present a graphical choice
        // for overlength line segment selection instead of auto-applying.
        const currentPlayer = this.gameState.currentPlayer;
        const player = this.gameState.players.find((p) => p.playerNumber === currentPlayer);
        const isHuman = player && player.type === 'human';

        if (isHuman && rewardCandidates.length > 1) {
          // Build segments for graphical selection
          const segments: LineRewardChoice['segments'] = rewardCandidates.map((m) => {
            const isCollapseAll =
              m.collapsedMarkers?.length === m.formedLines?.[0]?.positions?.length;
            return {
              optionId: m.id || `move-${rewardCandidates.indexOf(m)}`,
              positions: m.collapsedMarkers || [],
              isCollapseAll,
            };
          });

          // Build the choice with segment data for graphical rendering
          const choice: LineRewardChoice = {
            id: `line-reward-${Date.now()}`,
            gameId: this.gameState.id,
            playerNumber: currentPlayer,
            type: 'line_reward_option',
            prompt: 'Overlength Line - Choose which markers to collapse',
            options: rewardCandidates.map((m) => {
              const isCollapseAll =
                m.collapsedMarkers?.length === m.formedLines?.[0]?.positions?.length;
              return isCollapseAll
                ? ('option_1_collapse_all_and_eliminate' as const)
                : ('option_2_min_collapse_no_elimination' as const);
            }),
            moveIds: rewardCandidates.reduce(
              (acc, m) => {
                const isCollapseAll =
                  m.collapsedMarkers?.length === m.formedLines?.[0]?.positions?.length;
                const key = isCollapseAll
                  ? 'option_1_collapse_all_and_eliminate'
                  : 'option_2_min_collapse_no_elimination';
                acc[key] = m.id || `move-${rewardCandidates.indexOf(m)}`;
                return acc;
              },
              {} as Record<string, string>
            ),
            segments,
            linePositions: line.positions,
          };

          // Request choice from player via interaction handler
          const response = await this.interactionHandler.requestChoice(choice);

          // Find the selected move based on response
          // Response can come from segment click (optionId) or button click
          const selectedOptId =
            (response.selectedOption as { optionId?: string })?.optionId ||
            (response.selectedOption as string);

          // Try to match by optionId first (from segment click)
          let selectedMove = rewardCandidates.find((m) => m.id === selectedOptId);

          // If not found, try to match by option type (from button click)
          if (!selectedMove && typeof selectedOptId === 'string') {
            const isCollapseAllSelected = selectedOptId.includes('option_1');
            selectedMove = rewardCandidates.find((m) => {
              const isCollapseAll =
                m.collapsedMarkers?.length === m.formedLines?.[0]?.positions?.length;
              return isCollapseAll === isCollapseAllSelected;
            });
          }

          if (selectedMove) {
            moveToApply = selectedMove;
          } else {
            // Fallback to first candidate
            moveToApply = rewardCandidates[0];
          }
        } else {
          // AI player or only one option: auto-apply minimum collapse
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
          } else if (rewardCandidates.length > 0) {
            // Fallback: use first available reward candidate to prevent freeze
            moveToApply = rewardCandidates[0];
          }
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
      const outcome =
        moveToApply.type === 'process_line'
          ? applyProcessLineDecision(this.gameState, moveToApply)
          : applyChooseLineRewardDecision(this.gameState, moveToApply);
      const nextState = outcome.nextState;

      // Check if state actually changed
      if (hashGameState(nextState) === hashGameState(this.gameState)) {
        break;
      }

      this.gameState = nextState;

      // Track collapsed marker positions for sandbox-only visual cues. When
      // the shared helper reports explicit collapsedMarkers, prefer those;
      // otherwise fall back to the original line geometry when available.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing optional property from shared helper
      const outcomeAny = outcome as any;
      if (outcomeAny.collapsedMarkers && outcomeAny.collapsedMarkers.length > 0) {
        for (const pos of outcomeAny.collapsedMarkers as Position[]) {
          recentLineKeys.add(positionToString(pos));
        }
      } else if (line && line.positions && line.positions.length > 0) {
        for (const pos of line.positions) {
          recentLineKeys.add(positionToString(pos));
        }
      }

      // For automatic sandbox flows, immediately apply the cap elimination
      // when the shared helper reports a pending line-reward elimination
      // (exact-length lines and collapse-all rewards).
      if (outcome.pendingLineRewardElimination) {
        this.eliminateRingForLineReward(moveToApply.player);
      }

      // Record the canonical decision in history so that parity harnesses
      // can replay the exact same sequence into both engines.
      this.appendHistoryEntry(beforeState, moveToApply);
    }

    // Publish the accumulated highlights for the sandbox host. When no lines
    // were processed this call, this clears any previous highlight buffer.
    this._recentLineHighlightKeys = Array.from(recentLineKeys);
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
    const state = this.gameState;

    if (state.gameStatus !== 'active') {
      return false;
    }

    if (state.currentPhase !== 'ring_placement') {
      return false;
    }

    const player = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (!player || player.ringsInHand <= 0) {
      return false;
    }

    const count = Math.max(1, requestedCount);

    // Delegate legality (including caps + no-dead-placement) to the canonical
    // PlacementAggregate validator so sandbox semantics stay aligned with the
    // shared rules engine and backend RuleEngine.
    const action: PlaceRingAction = {
      type: 'PLACE_RING',
      playerId: state.currentPlayer,
      position,
      count,
    };
    const validation = validatePlacementAggregate(state, action);

    if (!validation.valid) {
      return false;
    }

    const beforeState = state;
    const key = positionToString(position);

    const placementMove: Move = {
      id: '',
      type: 'place_ring',
      player: state.currentPlayer,
      from: undefined,
      to: position,
      placementCount: count,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: beforeState.moveHistory.length + 1,
    } as Move;

    const outcome = applyPlacementMoveAggregate(beforeState, placementMove);

    if (hashGameState(outcome.nextState) === hashGameState(beforeState)) {
      return false;
    }

    this.gameState = {
      ...outcome.nextState,
      currentPhase: 'movement',
    };

    this._hasPlacedThisTurn = true;
    this._mustMoveFromStackKey = key;
    this._selectedStackKey = key;

    // Process automatic post-placement consequences (lines, territory, victory)
    // using the same helpers as the movement pipeline so placement-only turns
    // share the canonical processing order.
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
    _opts: { bypassNoDeadPlacement?: boolean } = {}
  ): Promise<boolean> {
    this.debugCheckpoint(`before-applyCanonicalMoveInternal-${move.type}`);
    const beforeState = this.getGameState();
    const beforeHash = hashGameState(beforeState);

    // ═══════════════════════════════════════════════════════════════════════
    // Orchestrator Adapter Delegation (permanently enabled)
    // ═══════════════════════════════════════════════════════════════════════
    // Delegate all rules logic to the shared orchestrator. This eliminates
    // duplicated logic and ensures sandbox and backend use identical rules processing.
    const changedByAdapter = await this.processMoveViaAdapter(move, beforeState);

    // Debug checkpoint after adapter processing
    this.debugCheckpoint(`after-applyCanonicalMoveInternal-adapter-${move.type}`);

    // Verify state actually changed
    const afterState = this.getGameState();
    const afterHash = hashGameState(afterState);
    if (changedByAdapter && beforeHash !== afterHash) {
      return true;
    }

    return false;
  }

  /**
   * Test-only helper: apply a single choose_territory_option Move (legacy
   * alias: process_territory_region) using the
   * same canonical pipeline as applyCanonicalMove, returning a boolean that
   * indicates whether the move changed state. This exists so RulesMatrix
   * territory scenarios can exercise Q23 preconditions against the sandbox
   * without going through the full turn/phase machinery.
   */
  private async applyCanonicalProcessTerritoryRegion(move: Move): Promise<boolean> {
    const canonicalType = normalizeMoveType(move.type);
    if (canonicalType !== 'choose_territory_option') {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalProcessTerritoryRegion: expected choose_territory_option (or legacy process_territory_region), got ${move.type}`
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
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
      'skip_capture',
      'process_line',
      'choose_line_option',
      'choose_territory_option',
      'eliminate_rings_from_stack',
      // Legacy replay compatibility
      'line_formation',
      'territory_claim',
      // Forced-elimination phase moves (canonical 7th phase). These are emitted
      // by the shared orchestrator for players who are blocked with stacks but
      // have no legal placement/movement/capture actions.
      'forced_elimination',
      // Meta-move: pie rule (swap colours after Player 1's first turn).
      // Handled by the shared orchestrator via validateSwapSidesMove/applySwapSidesIdentitySwap.
      'swap_sides',
      // Meta-move: skip remaining territory region decisions and advance turn.
      // Handled by the shared orchestrator which skips territory processing
      // when it detects this move type.
      'skip_territory_processing',
      // No-op actions: emitted when a player has no legal actions in a phase
      // (e.g., 0 rings in hand during ring_placement, no stacks during movement).
      // These advance the phase without changing board state.
      'no_placement_action',
      'no_movement_action',
      // RR-FIX-2025-12-13: Line/territory no-action moves emitted when the AI
      // is in line_processing or territory_processing with no interactive moves.
      'no_line_action',
      'no_territory_action',
    ];

    const canonicalType = normalizeMoveType(move.type);
    if (!supportedTypes.includes(canonicalType)) {
      throw new Error(`ClientSandboxEngine.applyCanonicalMove: unsupported move type ${move.type}`);
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
      // choose_territory_option moves before advancing (legacy alias: process_territory_region).
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

      // The orchestrator adapter already recorded the canonical Move into
      // moveHistory/history. Capture snapshots only so HistoryPlayback remains aligned.
      this.recordHistorySnapshotsOnly(beforeStateForHistory);
    }
  }

  /**
   * Replay helper: apply a canonical Move into the sandbox engine and always
   * record a history entry, even when the move does not change state.
   *
   * This is used by the /sandbox self-play replay path so that:
   * - moveHistory.length matches the recorder's total_moves, and
   * - the HistoryPlayback slider can scrub per recorded step, even for
   *   no-op bookkeeping moves that leave the board unchanged.
   *
   * Behaviour for state-changing moves mirrors applyCanonicalMove; for
   * no-op moves we skip any extra phase/turn advancement and simply record
   * a before/after history entry with identical snapshots.
   */
  public async applyCanonicalMoveForReplay(move: Move, nextMove?: Move | null): Promise<void> {
    const beforeStateForHistory = this.getGameState();

    // If game is already complete, still record the move in history for UI
    // purposes (history scrubbing) but don't actually apply it.
    if (this.gameState.gameStatus !== 'active') {
      this.appendHistoryEntry(beforeStateForHistory, move);
      return;
    }

    // Per RR-CANON-R075: In traceMode (canonical replay), all phases must be
    // visited with explicit moves. Skip auto-resolve and trust the recorded
    // move sequence to have correct phase transitions. In non-traceMode (legacy
    // recordings or live play), auto-resolve may still be needed.
    if (!this.traceMode) {
      await this.autoResolvePendingDecisionPhasesForReplay(move);
    }

    // If auto-resolve caused game completion, still record the move
    if (this.gameState.gameStatus !== 'active') {
      this.appendHistoryEntry(beforeStateForHistory, move);
      return;
    }

    const supportedTypes: Move['type'][] = [
      'place_ring',
      'skip_placement',
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
      'skip_capture',
      'process_line',
      'choose_line_option',
      'choose_territory_option',
      'eliminate_rings_from_stack',
      // Forced-elimination phase moves (canonical 7th phase). These appear in
      // canonical replay streams for blocked players and must be accepted by
      // the sandbox replayer so TS↔Python parity holds.
      'forced_elimination',
      // Meta-move: pie rule (swap colours after Player 1's first turn).
      // The shared orchestrator validates and applies this move.
      'swap_sides',
      // RR-CANON-R075 no-op markers: record that a phase was visited but
      // no action was possible. These are generated by Python's GameEngine
      // when a phase has no available actions. During replay they are
      // applied as history-only entries that don't modify board state.
      'no_territory_action',
      'no_line_action',
      'no_placement_action',
      'no_movement_action',
    ];

    const canonicalType = normalizeMoveType(move.type);
    if (!supportedTypes.includes(canonicalType)) {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalMoveForReplay: unsupported move type ${move.type}`
      );
    }

    const changed = await this.applyCanonicalMoveInternal(move, {
      bypassNoDeadPlacement: true,
    });

    // Debug: trace phase after orchestrator for move_stack parity investigation
    if (canonicalType === 'move_stack') {
      debugLog(
        SANDBOX_AI_STALL_DIAGNOSTICS,
        '[applyCanonicalMoveForReplay] POST-ORCHESTRATOR move_stack:',
        {
          movePlayer: move.player,
          moveTo: move.to ? positionToString(move.to) : null,
          currentPhase: this.gameState.currentPhase,
          currentPlayer: this.gameState.currentPlayer,
          gameStatus: this.gameState.gameStatus,
          changed,
        }
      );
    }

    if (changed) {
      // Phase transition fix: After place_ring, Python automatically advances
      // to movement phase for the same player. The TS orchestrator may leave
      // us in ring_placement, so we need to explicitly advance to movement.
      // This matches Python's post-placement semantics where the player who
      // placed a ring must then move the resulting stack.
      if (
        this.gameState.gameStatus === 'active' &&
        move.type === 'place_ring' &&
        this.gameState.currentPhase === 'ring_placement' &&
        this.gameState.currentPlayer === move.player
      ) {
        this.gameState.currentPhase = 'movement';
      }

      // In replay mode we want post-move snapshots to match the backend
      // GameReplayDB.get_state_at_move semantics, which expose the state
      // *after* any mandatory capture chains have been fully resolved. If
      // the shared orchestrator leaves the sandbox in a generic 'capture'
      // phase but there are no further capture segments available for the
      // current player, advance the turn immediately so the snapshot
      // reflects the next-turn phase (ring_placement or movement).
      //
      // IMPORTANT: For chain capture continuation, we must only check captures
      // from the LANDING POSITION of the just-applied capture, not all stacks.
      // This matches Python's chain_capture_state.current_position behavior.
      // For capture moves, check if chain continuation is required from landing position.
      // For non-capture moves that result in capture phase (e.g., move_stack that enables
      // a capture), let autoResolvePendingDecisionPhasesForReplay handle the phase
      // transition when the next move is applied - it knows whether the next recorded
      // move is a capture or not.
      //
      // PARITY FIX: In traceMode, do NOT manually advance after captures. The
      // orchestrator with skipTerritoryAutoResolve already handles the capture →
      // line_processing → territory_processing transitions correctly. Manual
      // advancement here would skip over line_processing and cause divergence
      // from Python's get_state_at_move which stays in the intermediate phase.
      if (
        !this.traceMode &&
        this.gameState.gameStatus === 'active' &&
        (this.gameState.currentPhase === 'capture' ||
          this.gameState.currentPhase === 'chain_capture') &&
        (move.type === 'overtaking_capture' || move.type === 'continue_capture_segment')
      ) {
        // Only check for continuation captures from the landing position
        // This matches Python's chain_capture_state.current_position behavior.
        const landingPosition = move.to;
        const continuationCaptures = this.enumerateCaptureSegmentsFrom(
          landingPosition,
          this.gameState.currentPlayer
        );
        if (continuationCaptures.length === 0) {
          this.advanceTurnAndPhaseForCurrentPlayer();
        }
      }

      // PARITY FIX: Same as above - skip manual territory advancement in traceMode.
      if (
        !this.traceMode &&
        this.gameState.gameStatus === 'active' &&
        this.gameState.currentPhase === 'territory_processing' &&
        (move.type === 'overtaking_capture' || move.type === 'continue_capture_segment')
      ) {
        const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
        const eligible = disconnected.filter((region) =>
          this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
        );

        if (eligible.length === 0) {
          this.advanceTurnAndPhaseForCurrentPlayer();
        }
      }

      // RR-CANON-R073: Mandatory phase transition after move_stack.
      // Per the canonical rules, after a non-capture movement, if legal capture
      // segments exist from the landing position, currentPhase MUST change to
      // 'capture'. This ensures parity with Python which transitions immediately.
      // IMPORTANT: Use move.player (the actual mover), not currentPlayer (which
      // may have already advanced to the next player by this point).
      if (
        this.gameState.gameStatus === 'active' &&
        this.gameState.currentPhase === 'movement' &&
        canonicalType === 'move_stack' &&
        move.to
      ) {
        const capturesFromLanding = this.enumerateCaptureSegmentsFrom(move.to, move.player);
        debugLog(
          SANDBOX_AI_STALL_DIAGNOSTICS,
          '[applyCanonicalMoveForReplay] RR-CANON-R073 capture check:',
          {
            movePlayer: move.player,
            moveTo: positionToString(move.to),
            capturesFound: capturesFromLanding.length,
            willTransition: capturesFromLanding.length > 0,
          }
        );
        if (capturesFromLanding.length > 0) {
          // Transition to capture phase per RR-CANON-R073
          this.gameState = {
            ...this.gameState,
            currentPhase: 'capture' as GamePhase,
          };
        }
      }

      // RR-CANON-R073: Mandatory phase transition after overtaking_capture.
      // After executing a capture segment, if additional legal captures exist
      // from the new landing position, transition to chain_capture phase.
      if (
        this.gameState.gameStatus === 'active' &&
        (this.gameState.currentPhase === 'capture' ||
          this.gameState.currentPhase === 'chain_capture') &&
        move.type === 'overtaking_capture' &&
        move.to
      ) {
        const chainCaptures = this.enumerateCaptureSegmentsFrom(move.to, move.player);
        if (chainCaptures.length > 0) {
          // Transition to chain_capture phase per RR-CANON-R073
          this.gameState = {
            ...this.gameState,
            currentPhase: 'chain_capture' as GamePhase,
          };
        }
      }

      // RR-CANON-R073: Mandatory phase transition after continue_capture_segment.
      // Same logic as overtaking_capture - if more captures exist, stay in chain_capture.
      if (
        this.gameState.gameStatus === 'active' &&
        this.gameState.currentPhase === 'chain_capture' &&
        move.type === 'continue_capture_segment' &&
        move.to
      ) {
        const moreCaptures = this.enumerateCaptureSegmentsFrom(move.to, move.player);
        if (moreCaptures.length > 0) {
          // Remain in chain_capture phase per RR-CANON-R073
          this.gameState = {
            ...this.gameState,
            currentPhase: 'chain_capture' as GamePhase,
          };
        } else {
          // RR-FIX-2026-01-12: Transition OUT of chain_capture when no more captures.
          // This mirrors the logic at lines 3934-3951 but ensures the phase is advanced
          // even if that earlier block was skipped (e.g., in traceMode or due to timing).
          // Without this, the phase can get stuck in chain_capture causing AI freeze.
          if (!this.traceMode) {
            this.advanceTurnAndPhaseForCurrentPlayer();
          }
        }
      }

      // Lookahead phase alignment: if we know the next move, use it to align
      // the current phase/player with what Python expects. This handles cases
      // where the TS orchestrator's phase transitions differ from Python's
      // (e.g., for decision phases like line_processing or territory_processing).
      //
      // PARITY FIX: In traceMode, skip lookahead alignment. Python's
      // get_state_at_move returns the state immediately after applying moves
      // without any lookahead-based phase advancement. The state after move N
      // should reflect what the engine produces, not what's needed for move N+1.
      // This ensures TS state snapshots match Python's at each move index.
      if (nextMove && this.gameState.gameStatus === 'active') {
        if (!this.traceMode) {
          await this.autoResolvePendingDecisionPhasesForReplay(nextMove);
        } else {
          // In traceMode, only resolve decision phases when the next move is a
          // turn-start move and we're still sitting in a decision phase. This
          // keeps replay tolerant to recorded placement/movement moves that
          // arrive after territory/line phases without introducing extra
          // automatic processing.
          const turnStartMoves: Move['type'][] = [
            'place_ring',
            'skip_placement',
            'move_stack',
            'overtaking_capture',
            'continue_capture_segment',
            'swap_sides',
            'eliminate_rings_from_stack',
          ];
          const inTurnStartPhase =
            this.gameState.currentPhase === 'ring_placement' ||
            this.gameState.currentPhase === 'movement';
          const canonicalNextMoveType = normalizeMoveType(nextMove.type);
          if (!inTurnStartPhase && turnStartMoves.includes(canonicalNextMoveType)) {
            await this.autoResolvePendingDecisionPhasesForReplay(nextMove);
          }
        }
      }

      // End-game phase completion: when this is the LAST move (no nextMove),
      // we may need to advance phases and check victory to match Python's
      // post-move semantics. In strict replay/trace mode (traceMode=true),
      // do NOT advance phases - leave the state exactly as the orchestrator
      // returned it to match Python's GameEngine.apply_move semantics. In
      // normal sandbox UX we still auto-process to a terminal shape.
      if (!nextMove && this.gameState.gameStatus === 'active') {
        // First, check victory immediately - this catches Early LPS (one player
        // has all material) without needing to advance phases further.
        this.checkAndApplyVictory();

        // PARITY FIX: In traceMode, do NOT advance phases after the last move.
        // Python's get_state_at_move returns the state immediately after
        // GameEngine.apply_move, without any additional phase advancement.
        // Advancing phases here would cause divergence (e.g., movement →
        // line_processing) that Python doesn't do.
        if (!this.traceMode) {
          const maxIterations = 50;
          let iterations = 0;
          while (this.gameState.gameStatus === 'active' && iterations < maxIterations) {
            const prevPhase = this.gameState.currentPhase;
            const prevPlayer = this.gameState.currentPlayer;

            // Normal sandbox UX: process pending decision phases to completion.
            if (this.gameState.currentPhase === 'territory_processing') {
              const resolved = await this.autoResolveOneTerritoryRegionForReplay();
              if (!resolved) {
                this.advanceTurnAndPhaseForCurrentPlayer();
              }
            } else if (this.gameState.currentPhase === 'line_processing') {
              const resolved = await this.autoResolveOneLineForReplay();
              if (!resolved) {
                this.advanceTurnAndPhaseForCurrentPlayer();
              }
            } else {
              this.advanceTurnAndPhaseForCurrentPlayer();
            }

            // Check victory after each phase advancement
            this.checkAndApplyVictory();

            if (
              this.gameState.currentPhase === prevPhase &&
              this.gameState.currentPlayer === prevPlayer
            ) {
              break;
            }
            iterations += 1;
          }
        }
      }

      // Align terminal metadata with backend: when the game ends, advance the
      // current player to the next turn holder so game_over snapshots match
      // Python's post-advance semantics used in parity harnesses.
      if (this.gameState.gameStatus !== 'active' && this.gameState.currentPhase === 'game_over') {
        const nextPlayer = this.getNextPlayerNumber(move.player);
        this.gameState = {
          ...this.gameState,
          currentPlayer: nextPlayer,
        };
      }

      // Note: The orchestrator adapter already adds the move to moveHistory, so we
      // skip that here to avoid duplicate entries.
      this.recordHistorySnapshotsOnly(beforeStateForHistory);
    } else {
      // No semantic change, but we still want a stable history step for
      // replay purposes. Record a history entry with identical before/after
      // snapshots so moveHistory and _stateSnapshots remain aligned with
      // the recorder's canonical move numbering.

      // Phase transition fix for no-change moves: Even if the orchestrator
      // reported no change, Python may have recorded a place_ring that we
      // need to honor by advancing to movement phase. This can happen when
      // the placement is into an already-occupied position (duplicate) or
      // when the orchestrator's change detection differs from Python's.
      if (
        this.gameState.gameStatus === 'active' &&
        move.type === 'place_ring' &&
        this.gameState.currentPhase === 'ring_placement' &&
        this.gameState.currentPlayer === move.player
      ) {
        this.gameState.currentPhase = 'movement';
      }

      // Even if the move didn't change state, we still need to run lookahead
      // to align phase/player with Python's expectations for the next move.
      if (nextMove && this.gameState.gameStatus === 'active') {
        const beforePhase = this.gameState.currentPhase;
        const beforePlayer = this.gameState.currentPlayer;
        await this.autoResolvePendingDecisionPhasesForReplay(nextMove);
        if (
          this.gameState.currentPhase !== beforePhase ||
          this.gameState.currentPlayer !== beforePlayer
        ) {
          debugLog(
            this.traceMode,
            `[lookahead-nochange] Advanced from ${beforePlayer}:${beforePhase} to ${this.gameState.currentPlayer}:${this.gameState.currentPhase} (nextMove=${nextMove.type} by p${nextMove.player})`
          );
        }
      }

      // End-game phase completion for no-change moves (same logic as changed branch).
      // PARITY FIX: In traceMode, skip phase advancement after last move.
      if (!nextMove && this.gameState.gameStatus === 'active') {
        // First, check victory immediately - this catches Early LPS.
        this.checkAndApplyVictory();

        // In traceMode, do NOT advance phases after the last move for parity with Python.
        if (!this.traceMode) {
          const maxIterations = 50;
          let iterations = 0;
          while (this.gameState.gameStatus === 'active' && iterations < maxIterations) {
            const prevPhase = this.gameState.currentPhase;
            const prevPlayer = this.gameState.currentPlayer;

            // Normal sandbox UX: auto-process remaining decisions.
            if (this.gameState.currentPhase === 'territory_processing') {
              const resolved = await this.autoResolveOneTerritoryRegionForReplay();
              if (!resolved) {
                this.advanceTurnAndPhaseForCurrentPlayer();
              }
            } else if (this.gameState.currentPhase === 'line_processing') {
              const resolved = await this.autoResolveOneLineForReplay();
              if (!resolved) {
                this.advanceTurnAndPhaseForCurrentPlayer();
              }
            } else {
              this.advanceTurnAndPhaseForCurrentPlayer();
            }

            // Check victory after each phase advancement
            this.checkAndApplyVictory();

            if (
              this.gameState.currentPhase === prevPhase &&
              this.gameState.currentPlayer === prevPlayer
            ) {
              break;
            }
            iterations += 1;
          }
        }
      }

      if (this.gameState.gameStatus !== 'active' && this.gameState.currentPhase === 'game_over') {
        const nextPlayer = this.getNextPlayerNumber(move.player);
        this.gameState = {
          ...this.gameState,
          currentPlayer: nextPlayer,
        };
      }

      // Even when the orchestrator reports no semantic change, the adapter
      // still records the move into moveHistory/history (normalized). Capture
      // snapshots only to keep HistoryPlayback aligned.
      this.recordHistorySnapshotsOnly(beforeStateForHistory);
    }
  }

  /**
   * Auto-resolve pending decision phases (territory_processing, line_processing)
   * before attempting to apply a move that expects a different phase.
   *
   * The Python AI engine auto-handled these phases internally without recording
   * explicit decision moves, so during replay we must auto-resolve them to
   * maintain phase alignment with the recorded move sequence.
   */
  private async autoResolvePendingDecisionPhasesForReplay(nextMove: Move): Promise<void> {
    // Only auto-resolve if the next move expects a turn-start phase (placement/movement)
    // and we're currently stuck in a decision phase
    const turnStartPhases: GamePhase[] = ['ring_placement', 'movement'];
    const turnStartMoveTypes: Move['type'][] = [
      'place_ring',
      'skip_placement',
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
      // Swap-sides is offered at the start of Player 2's interactive turn.
      'swap_sides',
      // Forced elimination occurs at the start of a blocked player's turn.
      'eliminate_rings_from_stack',
    ];

    // Decision-phase move types that indicate what phase we should be in
    // Include no-op markers (no_line_action, no_territory_action) so auto-resolve
    // can transition to the expected phase when Python recorded a no-op marker
    const lineProcessingMoveTypes: Move['type'][] = [
      'process_line',
      'choose_line_option',
      'no_line_action',
      // Legacy replay compatibility
      'line_formation',
    ];
    const territoryProcessingMoveTypes: Move['type'][] = [
      'choose_territory_option',
      'no_territory_action',
      // Legacy replay compatibility
      'territory_claim',
    ];

    // Handle ring_placement → movement skip: if we're in ring_placement but the
    // next move is a movement/capture from the SAME player (not place_ring or
    // skip_placement), Python must have skipped placement due to no legal placements.
    // We should advance directly to movement phase to match.
    const movementMoveTypes: Move['type'][] = [
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
    ];
    const canonicalNextMoveType = normalizeMoveType(nextMove.type);
    if (
      this.gameState.currentPhase === 'ring_placement' &&
      movementMoveTypes.includes(canonicalNextMoveType) &&
      nextMove.player === this.gameState.currentPlayer
    ) {
      // Python recorded a movement move while player was in ring_placement.
      // This means Python auto-skipped to movement (no legal placements).
      this.gameState.currentPhase = 'movement';
    }

    // Capture move types for phase alignment (reserved for future capture phase handling)
    const _captureMoveTypes: Move['type'][] = ['overtaking_capture', 'continue_capture_segment'];

    // NOTE: movement → capture alignment has been moved to applyCanonicalMoveForReplay
    // because it should only apply after move_stack, not after place_ring.
    // See the post-move_stack handling in applyCanonicalMoveForReplay for details.

    if (!turnStartMoveTypes.includes(canonicalNextMoveType)) {
      // The next move is a decision-phase move. Check if we're in a mismatched phase
      // and need to transition. This handles cases like TS being in 'capture' phase
      // when Python expects 'line_processing' (because line detection takes priority).
      const currentPhase = this.gameState.currentPhase as string;
      if (
        lineProcessingMoveTypes.includes(canonicalNextMoveType) &&
        currentPhase !== 'line_processing'
      ) {
        // Need to transition to line_processing. Advance turn until we get there.
        const maxAdvances = 10;
        let advances = 0;
        while (
          this.gameState.gameStatus === 'active' &&
          (this.gameState.currentPhase as string) !== 'line_processing' &&
          advances < maxAdvances
        ) {
          this.advanceTurnAndPhaseForCurrentPlayer();
          advances += 1;
        }
      } else if (
        territoryProcessingMoveTypes.includes(canonicalNextMoveType) &&
        currentPhase !== 'territory_processing'
      ) {
        // Need to transition to territory_processing
        const maxAdvances = 10;
        let advances = 0;
        while (
          this.gameState.gameStatus === 'active' &&
          (this.gameState.currentPhase as string) !== 'territory_processing' &&
          advances < maxAdvances
        ) {
          this.advanceTurnAndPhaseForCurrentPlayer();
          advances += 1;
        }
      }

      // Player alignment for decision-phase moves: if we're in the correct phase
      // but with the wrong player, set the player to match Python's expectation.
      // This handles cases where the TS orchestrator advances to the next player
      // after movement, but Python keeps the same player for territory processing
      // (because the moving player caused the disconnection and should process it).
      if (
        this.gameState.gameStatus === 'active' &&
        territoryProcessingMoveTypes.includes(canonicalNextMoveType) &&
        this.gameState.currentPhase === 'territory_processing' &&
        nextMove.player !== this.gameState.currentPlayer
      ) {
        this.gameState.currentPlayer = nextMove.player;
      }

      // RR-CANON-R121: Player alignment for line_processing moves.
      // Per the canonical rules, the "moving player" who triggered line detection
      // should process their lines. The TS orchestrator may advance to the next
      // player before line_processing, so we align currentPlayer to match the
      // recorded move's player.
      if (
        this.gameState.gameStatus === 'active' &&
        lineProcessingMoveTypes.includes(canonicalNextMoveType) &&
        this.gameState.currentPhase === 'line_processing' &&
        nextMove.player !== this.gameState.currentPlayer
      ) {
        this.gameState.currentPlayer = nextMove.player;
      }

      // After alignment (or if already aligned), let the decision move proceed
      return;
    }

    // Safety limit to prevent infinite loops
    const maxIterations = 100;
    let iterations = 0;

    while (
      this.gameState.gameStatus === 'active' &&
      !turnStartPhases.includes(this.gameState.currentPhase) &&
      iterations < maxIterations
    ) {
      iterations += 1;

      if (this.gameState.currentPhase === 'territory_processing') {
        // Check if there are pending territory regions to process
        const regions = findDisconnectedRegionsOnBoard(this.gameState.board);
        const eligible = regions.filter((region) =>
          this.canProcessDisconnectedRegion(
            region.spaces,
            this.gameState.currentPlayer,
            this.gameState.board
          )
        );

        if (eligible.length === 0) {
          // No pending regions - safe to advance phase/turn, unless the
          // next recorded move is itself a territory decision. In traceMode
          // we trust the recorded move sequence over TS detection.
          if (
            this.traceMode &&
            (canonicalNextMoveType === 'choose_territory_option' ||
              canonicalNextMoveType === 'eliminate_rings_from_stack' ||
              nextMove.type === 'territory_claim')
          ) {
            break;
          }
          this.advanceTurnAndPhaseForCurrentPlayer();
        } else if (this.traceMode) {
          // PARITY FIX: In traceMode, if there are pending regions, do NOT
          // auto-process them. Python records territory decisions as explicit
          // moves (choose_territory_option; legacy alias: process_territory_region),
          // so we must let those recorded
          // moves arrive in sequence. Break here and wait for them.
          break;
        } else {
          const resolved = await this.autoResolveOneTerritoryRegionForReplay();
          if (!resolved) {
            // No more regions to process, but still in territory_processing
            // This shouldn't happen, but advance turn to recover
            this.advanceTurnAndPhaseForCurrentPlayer();
          }
        }
      } else if (this.gameState.currentPhase === 'line_processing') {
        // Check if there are pending lines to process
        const lineMoves = enumerateProcessLineMoves(this.gameState, this.gameState.currentPlayer);

        if (lineMoves.length === 0) {
          // No pending lines - safe to advance phase/turn, unless the next
          // recorded move is itself a line-processing decision. In traceMode
          // we trust the recorded move sequence over TS detection.
          if (
            this.traceMode &&
            (canonicalNextMoveType === 'process_line' ||
              canonicalNextMoveType === 'choose_line_option' ||
              nextMove.type === 'line_formation')
          ) {
            break;
          }
          this.advanceTurnAndPhaseForCurrentPlayer();
        } else if (this.traceMode) {
          // PARITY FIX: In traceMode, if there are pending lines, do NOT
          // auto-process them. Python records line decisions as explicit
          // moves (process_line, choose_line_option, legacy line_formation), so we must let those
          // recorded moves arrive in sequence. Break here and wait for them.
          break;
        } else {
          const phaseBefore = this.gameState.currentPhase;
          const resolved = await this.autoResolveOneLineForReplay();
          if (!resolved) {
            // No more lines to process, advance turn
            this.advanceTurnAndPhaseForCurrentPlayer();
          } else if (this.gameState.currentPhase === phaseBefore) {
            // autoResolveOneLineForReplay processed something but phase didn't change.
            // This can happen when line processing is already complete but the
            // orchestrator leaves us in line_processing. Force advance.
            this.advanceTurnAndPhaseForCurrentPlayer();
          }
        }
      } else if (
        this.gameState.currentPhase === 'capture' ||
        this.gameState.currentPhase === 'chain_capture'
      ) {
        // Capture/chain_capture phase handling: Python games DO record capture moves,
        // so only advance if the next move is NOT a capture from the current player.
        // This fixes parity divergence where TS was incorrectly advancing before
        // the capture sequence was applied.
        const isNextMoveCapture =
          (nextMove.type === 'overtaking_capture' ||
            nextMove.type === 'continue_capture_segment') &&
          nextMove.player === this.gameState.currentPlayer;
        if (isNextMoveCapture) {
          // The next move is a valid capture - let it be applied
          break;
        }
        // Next move is from a different player or phase - advance
        this.advanceTurnAndPhaseForCurrentPlayer();
      } else {
        // Unknown phase, try to advance
        this.advanceTurnAndPhaseForCurrentPlayer();
      }
    }

    if (iterations >= maxIterations) {
      console.warn(
        '[autoResolvePendingDecisionPhasesForReplay] Max iterations reached, possible infinite loop',
        { currentPhase: this.gameState.currentPhase, nextMoveType: nextMove.type }
      );
    }

    // Handle player mismatch: if the next move is from a different player, we
    // may need to advance until we reach that player. However, RR-CANON-R208
    // requires that line_processing and territory_processing be fully resolved
    // before re-entering a new player's interactive turn. When there are still
    // pending line or Territory decisions for the current player, we must NOT
    // auto-rotate to align with a turn-start move from the recording; instead,
    // we leave the state in the decision phase so canonical validation can
    // surface any illegal historical moves.
    if (
      this.gameState.gameStatus === 'active' &&
      turnStartMoveTypes.includes(canonicalNextMoveType) &&
      nextMove.player !== this.gameState.currentPlayer
    ) {
      let hasPendingTerritory = false;
      let hasPendingLines = false;

      if (this.gameState.currentPhase === 'territory_processing') {
        const regions = findDisconnectedRegionsOnBoard(this.gameState.board);
        const eligible = regions.filter((region) =>
          this.canProcessDisconnectedRegion(
            region.spaces,
            this.gameState.currentPlayer,
            this.gameState.board
          )
        );
        hasPendingTerritory = eligible.length > 0;
      } else if (this.gameState.currentPhase === 'line_processing') {
        const lineMoves = enumerateProcessLineMoves(this.gameState, this.gameState.currentPlayer);
        hasPendingLines = lineMoves.length > 0;
      }

      if (!hasPendingTerritory && !hasPendingLines) {
        const maxPlayerAdvances = this.gameState.players.length * 3;
        let advances = 0;
        while (
          this.gameState.gameStatus === 'active' &&
          nextMove.player !== this.gameState.currentPlayer &&
          advances < maxPlayerAdvances
        ) {
          this.advanceTurnAndPhaseForCurrentPlayer();
          advances += 1;
        }
      }
    }

    // Final ring_placement → movement fix: After all player advancement is done,
    // check if we're now in ring_placement but the next move expects movement.
    // This handles the case where advanceTurnAndPhaseForCurrentPlayer puts us in
    // ring_placement but Python auto-skipped to movement (no legal placements).
    if (
      this.gameState.gameStatus === 'active' &&
      this.gameState.currentPhase === 'ring_placement' &&
      movementMoveTypes.includes(nextMove.type) &&
      nextMove.player === this.gameState.currentPlayer
    ) {
      this.gameState.currentPhase = 'movement';
    }
  }

  /**
   * Auto-process one territory region during replay.
   * Returns true if a region was processed, false if none available.
   */
  private async autoResolveOneTerritoryRegionForReplay(): Promise<boolean> {
    const state = this.gameState;
    const regions = findDisconnectedRegionsOnBoard(state.board);
    const eligible = regions.filter((region) =>
      this.canProcessDisconnectedRegion(region.spaces, state.currentPlayer, state.board)
    );

    if (eligible.length === 0) {
      return false;
    }

    // Process the first eligible region
    const region = eligible[0];
    const move: Move = {
      id: `auto-replay-territory-${Date.now()}`,
      type: 'choose_territory_option',
      player: state.currentPlayer,
      to: region.spaces[0],
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: state.moveHistory.length + 1,
    };

    // Apply via the orchestrator adapter
    const result = await this.processMoveViaAdapter(move, state);

    // Handle any self-elimination that results from territory processing
    if (
      result &&
      this.gameState.gameStatus === 'active' &&
      this.gameState.currentPhase === 'territory_processing'
    ) {
      // Check if elimination is required
      const elimMoves = enumerateTerritoryEliminationMoves(
        this.gameState,
        this.gameState.currentPlayer
      );
      if (elimMoves.length > 0) {
        // Auto-apply the first elimination option
        const elimMove = elimMoves[0];
        await this.processMoveViaAdapter(elimMove, this.gameState);
      }
    }

    return true;
  }

  /**
   * Auto-process one line during replay.
   * Returns true if a line was processed, false if none available.
   */
  private async autoResolveOneLineForReplay(): Promise<boolean> {
    const state = this.gameState;
    const lineMoves = enumerateProcessLineMoves(state, state.currentPlayer);

    if (lineMoves.length === 0) {
      return false;
    }

    // Process the first available line move
    const lineMove = lineMoves[0];
    await this.processMoveViaAdapter(lineMove, state);

    // Handle line reward choice if needed
    if (
      this.gameState.gameStatus === 'active' &&
      this.gameState.currentPhase === 'line_processing'
    ) {
      const currentPlayer = this.gameState.currentPlayer;
      const playerLines = findLinesForPlayer(this.gameState.board, currentPlayer);
      const rewardMoves: Move[] = [];
      playerLines.forEach((_line, index) => {
        rewardMoves.push(...enumerateChooseLineRewardMoves(this.gameState, currentPlayer, index));
      });
      if (rewardMoves.length > 0) {
        // Auto-select first reward option
        const rewardMove = rewardMoves[0];
        await this.processMoveViaAdapter(rewardMove, this.gameState);
      }
    }

    return true;
  }
}

// Test-only: attach a lightweight board-invariant helper to the prototype so
// invariant tests can exercise internal board sanity checks without expanding
// the public class surface for production code.
(ClientSandboxEngine.prototype as unknown as Record<string, unknown>).assertBoardInvariants =
  function (this: ClientSandboxEngine, context: string): void {
    const board: BoardState = (this as unknown as { gameState: GameState }).gameState.board;
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

    console.error(message);

    if (isTestEnvironment()) {
      throw new Error(message);
    }
  };
