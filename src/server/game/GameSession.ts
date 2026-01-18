import { Server as SocketIOServer } from 'socket.io';
import type { Move as PrismaMove } from '@prisma/client';
import { GameEngine } from './GameEngine';
import { RulesBackendFacade, RulesResult } from './RulesBackendFacade';
import { PlayerInteractionManager } from './PlayerInteractionManager';
import { WebSocketInteractionHandler } from './WebSocketInteractionHandler';
import { DelegatingInteractionHandler } from './DelegatingInteractionHandler';
import { AIInteractionHandler } from './ai/AIInteractionHandler';
import { globalAIEngine, AIDiagnostics } from './ai/AIEngine';
import { getOrCreateAIUser } from '../services/AIUserService';
import { PythonRulesClient } from '../services/PythonRulesClient';
import { GamePersistenceService } from '../services/GamePersistenceService';
import { gameRecordRepository } from '../services/GameRecordRepository';
import { GameOutcome, FinalScore } from '../../shared/types/gameRecord';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import { getMetricsService } from '../services/MetricsService';
import { orchestratorRollout } from '../services/OrchestratorRolloutService';
import { config } from '../config';
import {
  applyDecisionPhaseFixtureIfNeeded,
  type DecisionPhaseFixtureMetadata,
} from './testFixtures/decisionPhaseFixtures';
import type { AuthenticatedSocket } from '../websocket/server';
import {
  Move,
  Player,
  GameState,
  Position,
  AIProfile,
  BOARD_CONFIGS,
  TimeControl,
  GameResult,
  GamePhase,
  MoveType,
  PlayerChoiceType,
  LineInfo,
  Territory,
  AIControlMode,
  AITacticType,
} from '../../shared/types/game';
import type { GameStatus as PrismaGameStatus } from '@prisma/client';
import type {
  DecisionAutoResolvedMeta,
  DecisionPhaseTimeoutWarningPayload,
  DecisionPhaseTimedOutPayload,
} from '../../shared/types/websocket';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import type { LocalAIRng } from '../../shared/engine';
import { serializeBoardState } from '../../shared/engine/contracts/serialization';
import { SeededRNG } from '../../shared/utils/rng';
import { GameSessionStatus, deriveGameSessionStatus } from '../../shared/stateMachines/gameSession';
import {
  AIRequestState,
  idleAIRequest,
  markQueued,
  markInFlight,
  markFallbackLocal,
  markCompleted,
  markTimedOut,
  markFailed,
  markCanceled,
  isCancelable,
  isDeadlineExceeded,
  AIRequestCancelReason,
} from '../../shared/stateMachines/aiRequest';
import { runWithTimeout } from '../../shared/utils/timeout';
import { createCancellationSource, type CancellationToken } from '../../shared/utils/cancellation';
import { getAIServiceClient } from '../services/AIServiceClient';

/**
 * Aggregated AI quality diagnostics for a session
 */
export interface SessionAIDiagnostics {
  rulesServiceFailureCount: number;
  rulesShadowErrorCount: number;
  aiServiceFailureCount: number;
  aiFallbackMoveCount: number;
  aiQualityMode: 'normal' | 'fallbackLocalAI' | 'rulesServiceDegraded';
}

/**
 * Position with optional z coordinate that may be explicitly undefined
 * (compatible with Zod schema output under exactOptionalPropertyTypes).
 */
interface PositionWithOptionalZ {
  x: number;
  y: number;
  z?: number | undefined;
}

/**
 * Player move data payload received from WebSocket.
 * The position types are aligned with the Zod MoveSchema output.
 */
interface PlayerMoveData {
  position?:
    | string
    | { from?: PositionWithOptionalZ; to: PositionWithOptionalZ }
    | PositionWithOptionalZ;
  moveType: MoveType;
}

/**
 * Deserialized structure of the game.gameState JSON field.
 * This represents configuration stored at game creation time.
 */
interface PersistedGameStateSnapshot {
  aiOpponents?: {
    count: number;
    difficulty: number[];
    mode?: AIControlMode;
    aiType?: AITacticType;
    aiTypes?: AITacticType[];
  };
  rulesOptions?: GameState['rulesOptions'];
  fixture?: DecisionPhaseFixtureMetadata;
}

/**
 * Move type with optional decision-phase metadata attached during auto-resolution.
 */
type MoveWithAutoResolveMeta = Move & { decisionAutoResolved?: DecisionAutoResolvedMeta };

/**
 * GameSession manages a single game's lifecycle including:
 * - Game state management via GameEngine
 * - WebSocket communication
 * - AI turn handling with state machine tracking
 * - Player interaction handling
 * - Session status projection
 */
export class GameSession {
  public readonly gameId: string;
  private io: SocketIOServer;
  private gameEngine!: GameEngine;
  private rulesFacade!: RulesBackendFacade;
  private interactionManager!: PlayerInteractionManager;
  private wsHandler!: WebSocketInteractionHandler;
  private pythonRulesClient: PythonRulesClient;
  private userSockets: Map<string, string>; // userId -> socketId

  // Session status projection (derived from GameState)
  private sessionStatus: GameSessionStatus | null = null;

  // AI request state machine for current/last AI turn
  private aiRequestState: AIRequestState = idleAIRequest;

  // AbortController for canceling in-flight AI requests
  private aiAbortController: AbortController | null = null;

  // Session-scoped cancellation source used to cooperatively cancel
  // long-lived async operations such as AI requests when the session
  // is terminated.
  private readonly sessionCancellationSource = createCancellationSource();

  // Aggregated diagnostics
  private diagnosticsSnapshot: SessionAIDiagnostics = {
    rulesServiceFailureCount: 0,
    rulesShadowErrorCount: 0,
    aiServiceFailureCount: 0,
    aiFallbackMoveCount: 0,
    aiQualityMode: 'normal',
  };

  // Decision phase timeout tracking (per-session, per-active decision)
  private decisionTimeoutDeadlineMs: number | null = null;
  private decisionTimeoutPhase: GamePhase | null = null;
  private decisionTimeoutPlayer: number | null = null;
  private decisionTimeoutChoiceType: PlayerChoiceType | null = null;
  private decisionTimeoutChoiceKind: DecisionAutoResolvedMeta['choiceKind'] | null = null;
  private decisionTimeoutWarningHandle: NodeJS.Timeout | null = null;
  private decisionTimeoutHandle: NodeJS.Timeout | null = null;

  // Fixture metadata for test games (used for timeout overrides)
  private fixtureMetadata: DecisionPhaseFixtureMetadata | null = null;

  // Default AI request timeout (can be overridden via config)
  private readonly aiRequestTimeoutMs: number;

  // AI watchdog timer to detect stalled turns
  private aiWatchdogHandle: NodeJS.Timeout | null = null;
  private lastAITurnCheck = 0;

  // Mapping from playerId (UUID) to playerNumber for move replay
  private playerIdToNumber: Map<string, number> = new Map();

  // Lock callback for protecting concurrent state modifications (provided by GameSessionManager)
  private readonly withLock: <T>(operation: () => Promise<T>) => Promise<T>;

  constructor(
    gameId: string,
    io: SocketIOServer,
    pythonRulesClient: PythonRulesClient,
    userSockets: Map<string, string>,
    withLock?: <T>(operation: () => Promise<T>) => Promise<T>
  ) {
    this.gameId = gameId;
    this.io = io;
    this.pythonRulesClient = pythonRulesClient;
    this.userSockets = userSockets;
    this.aiRequestTimeoutMs = config.aiService?.requestTimeoutMs ?? 30000;
    // If no lock function provided, execute operations directly (fallback for tests)
    this.withLock = withLock ?? ((op) => op());
  }

  public async initialize(): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    // Load game from database
    const game = await prisma.game.findUnique({
      where: { id: this.gameId },
      include: {
        player1: { select: { id: true, username: true } },
        player2: { select: { id: true, username: true } },
        player3: { select: { id: true, username: true } },
        player4: { select: { id: true, username: true } },
        moves: {
          orderBy: { moveNumber: 'asc' },
        },
      },
    });

    if (!game) {
      throw new Error('Game not found');
    }

    // Optional AI opponents and per-game rules configuration (e.g., swap rule).
    // The gameState field is stored as JSON in Prisma and deserialized here.
    const rawGameState = game.gameState;
    const gameStateSnapshot: PersistedGameStateSnapshot =
      typeof rawGameState === 'string'
        ? (JSON.parse(rawGameState) as PersistedGameStateSnapshot)
        : ((rawGameState || {}) as PersistedGameStateSnapshot);
    const aiOpponents = gameStateSnapshot.aiOpponents;
    const rulesOptions = gameStateSnapshot.rulesOptions;

    // Create players array
    const players: Player[] = [];
    const boardConfig = BOARD_CONFIGS[game.boardType as keyof typeof BOARD_CONFIGS];
    const effectiveRingsPerPlayer = rulesOptions?.ringsPerPlayer ?? boardConfig.ringsPerPlayer;
    const initialTimeMs =
      typeof game.timeControl === 'string'
        ? (JSON.parse(game.timeControl).initialTime as number) * 1000 // Convert seconds to ms
        : 600000;

    if (game.player1) {
      players.push(this.createPlayer(game.player1, 1, effectiveRingsPerPlayer, initialTimeMs));
      this.playerIdToNumber.set(game.player1.id, 1);
    }
    if (game.player2) {
      players.push(this.createPlayer(game.player2, 2, effectiveRingsPerPlayer, initialTimeMs));
      this.playerIdToNumber.set(game.player2.id, 2);
    }
    if (game.player3) {
      players.push(this.createPlayer(game.player3, 3, effectiveRingsPerPlayer, initialTimeMs));
      this.playerIdToNumber.set(game.player3.id, 3);
    }
    if (game.player4) {
      players.push(this.createPlayer(game.player4, 4, effectiveRingsPerPlayer, initialTimeMs));
      this.playerIdToNumber.set(game.player4.id, 4);
    }

    if (aiOpponents && aiOpponents.count > 0) {
      const startingNumber = players.length + 1;
      const maxSlots = game.maxPlayers ?? 2;
      const aiCount = Math.min(aiOpponents.count, maxSlots - players.length);

      for (let i = 0; i < aiCount; i++) {
        const playerNumber = startingNumber + i;
        const difficulty = aiOpponents.difficulty?.[i] ?? 5;
        // Use per-player AI type from aiTypes array if available, falling back to shared aiType
        const aiType = aiOpponents.aiTypes?.[i] ?? aiOpponents.aiType;
        const aiProfile: AIProfile = {
          difficulty,
          mode: aiOpponents.mode ?? 'service',
          ...(aiType && { aiType }),
        };

        players.push({
          id: `ai-${this.gameId}-${playerNumber}`,
          username: `AI (Level ${difficulty})`,
          playerNumber,
          type: 'ai',
          isReady: true,
          timeRemaining: initialTimeMs,
          ringsInHand: effectiveRingsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: difficulty,
          aiProfile,
        });

        try {
          globalAIEngine.createAIFromProfile(playerNumber, aiProfile);
        } catch (err) {
          logger.error('Failed to configure AI player', {
            gameId: this.gameId,
            playerNumber,
            difficulty,
            error: (err as Error).message,
          });
        }
      }
    }

    // Create time control
    let timeControl: TimeControl;
    if (typeof game.timeControl === 'string') {
      timeControl = JSON.parse(game.timeControl) as TimeControl;
    } else if (game.timeControl && typeof game.timeControl === 'object') {
      timeControl = game.timeControl as unknown as TimeControl;
    } else {
      timeControl = { type: 'rapid', initialTime: 600000, increment: 0 };
    }

    // Setup interaction handlers
    const getTargetForPlayer = (playerNumber: number): string | undefined => {
      const player = players.find((p) => p.playerNumber === playerNumber);
      if (!player) return undefined;
      return this.userSockets.get(player.id);
    };

    this.wsHandler = new WebSocketInteractionHandler(
      this.io,
      this.gameId,
      getTargetForPlayer,
      30_000
    );

    // RR-FIX-2026-01-18: Broadcast intermediate game state before showing choice UI.
    // This ensures the board updates visually (e.g., piece moved) while the player
    // is deciding (e.g., which territory to claim).
    this.wsHandler.onBeforeChoice = async (choice) => {
      // Use the intermediate state from the choice if available, otherwise fall back
      // to the current engine state (which may be stale during decision processing)
      const intermediateState = choice.intermediateState ?? this.gameEngine?.getGameState();
      if (!intermediateState) {
        logger.warn('No intermediate state available for choice broadcast', {
          gameId: this.gameId,
          choiceType: choice.type,
        });
        return;
      }

      logger.info('Broadcasting intermediate state before choice', {
        gameId: this.gameId,
        choiceType: choice.type,
        playerNumber: choice.playerNumber,
        hasIntermediateState: !!choice.intermediateState,
        moveCount: intermediateState.moveHistory?.length,
      });

      // Broadcast the intermediate state to all clients in the room
      const room = this.io.sockets.adapter.rooms.get(this.gameId);
      if (room) {
        for (const socketId of room) {
          const socket = this.io.sockets.sockets.get(socketId) as AuthenticatedSocket | undefined;
          if (!socket) continue;

          // RR-FIX-2026-01-18: During intermediate state broadcast (before a choice),
          // we should NOT send valid moves because:
          // 1. The player is about to be prompted for a decision, not a game move
          // 2. The engine's internal state (mustMoveFromStackKey) may not match the
          //    intermediate state, causing incorrect valid moves to be calculated
          // Send empty array to prevent stale/incorrect moves from being displayed.
          const validMoves: Move[] = [];

          const transportState = {
            ...intermediateState,
            board: serializeBoardState(intermediateState.board),
          };

          const payload = {
            type: 'game_update' as const,
            data: {
              gameId: this.gameId,
              gameState: transportState as unknown as GameState,
              validMoves,
            },
            timestamp: new Date().toISOString(),
          };

          socket.emit('game_state', payload);
        }
      }
    };

    const aiHandler = new AIInteractionHandler(this.sessionCancellationSource.token);
    const delegatingHandler = new DelegatingInteractionHandler(
      this.wsHandler,
      aiHandler,
      (playerNumber: number) => {
        const player = players.find((p) => p.playerNumber === playerNumber);
        return player?.type ?? 'human';
      }
    );

    this.interactionManager = new PlayerInteractionManager(delegatingHandler);

    // Derive the per-game RNG seed from the persisted Game row when available.
    // This keeps backend GameState.rngSeed aligned with the database and the
    // Python AI service, which both treat this seed as the canonical RNG root.
    const rngSeed = typeof game.rngSeed === 'number' ? game.rngSeed : undefined;

    // Create game engine
    this.gameEngine = new GameEngine(
      this.gameId,
      game.boardType as keyof typeof BOARD_CONFIGS,
      players,
      timeControl,
      game.isRated,
      this.interactionManager,
      // Use the persisted per-game RNG seed when available so backend
      // GameState.rngSeed matches the database and AI service expectations.
      rngSeed,
      rulesOptions
    );

    // Decide which engine path to use for this session based on
    // orchestrator rollout feature flags and user targeting.
    this.configureEngineSelection(players);

    this.gameEngine.enableMoveDrivenDecisionPhases();

    // Replay moves
    for (const move of game.moves) {
      this.replayMove(move);
    }

    // In non-production environments, apply any configured decision-phase
    // fixture overlays after historical moves have been replayed. This
    // allows specialized test fixtures to start directly in a decision
    // phase (for example line_processing) without affecting production
    // game lifecycles.
    if (config.isTest || config.isDevelopment) {
      const fixtureApplied = applyDecisionPhaseFixtureIfNeeded(this.gameEngine, gameStateSnapshot);
      if (fixtureApplied && gameStateSnapshot.fixture) {
        this.fixtureMetadata = gameStateSnapshot.fixture as DecisionPhaseFixtureMetadata;
      }
    }

    this.rulesFacade = new RulesBackendFacade(this.gameEngine, this.pythonRulesClient);

    // Auto-start logic - handle both waiting games AND active AI games that need initialization
    // AI games are created with status 'active' (see game.ts), but still need startGame() and AI turn
    const hasAIPlayers = players.some((p) => p.type === 'ai');
    const needsInitialization =
      game.status === 'waiting' || (game.status === 'active' && hasAIPlayers);

    // Diagnostic: log auto-start decision factors
    logger.info('Auto-start check', {
      gameId: this.gameId,
      dbStatus: game.status,
      hasAIPlayers,
      needsInitialization,
      playerCount: players.length,
      maxPlayers: game.maxPlayers,
      playerTypes: players.map((p) => ({ num: p.playerNumber, type: p.type })),
    });

    if (needsInitialization && players.length >= (game.maxPlayers ?? 2)) {
      const allReady = players.every((p) => p.isReady);
      if (allReady) {
        try {
          // Only update DB status if transitioning from waiting (AI games are already active)
          if (game.status === 'waiting') {
            await prisma.game.update({
              where: { id: this.gameId },
              data: {
                status: 'active' as PrismaGameStatus,
                startedAt: new Date(),
              },
            });
          }
          const startGameResult = this.gameEngine.startGame();
          const afterStartState = this.gameEngine.getGameState();
          logger.info('Game started during init', {
            gameId: this.gameId,
            startGameResult,
            gameStatus: afterStartState.gameStatus,
            currentPlayer: afterStartState.currentPlayer,
            currentPhase: afterStartState.currentPhase,
          });

          // Trigger AI turn if the first player is AI
          await this.maybePerformAITurn();
        } catch (err) {
          logger.error('Failed to auto-start game', {
            gameId: this.gameId,
            error: (err as Error).message,
          });
        }
      }
    }

    // Compute initial session status
    this.recomputeSessionStatus();

    // Schedule initial decision-phase timeout (if applicable) based on
    // the reconstructed game state. This ensures that games resumed in a
    // decision phase – including test fixtures – get appropriate timeout
    // handling even before any new moves are made.
    this.scheduleDecisionPhaseTimeout(this.gameEngine.getGameState());

    // Start AI watchdog to detect and recover from stalled AI turns
    this.startAIWatchdog();

    logger.info('GameSession initialized', {
      gameId: this.gameId,
      status: this.sessionStatus?.kind,
      playerCount: players.length,
    });
  }

  private createPlayer(
    dbPlayer: { id: string; username: string | null },
    playerNumber: number,
    ringsPerPlayer: number,
    initialTimeMs: number
  ): Player {
    return {
      id: dbPlayer.id,
      username: dbPlayer.username ?? `Player ${playerNumber}`,
      playerNumber,
      type: 'human',
      isReady: true,
      timeRemaining: initialTimeMs,
      ringsInHand: ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    };
  }

  /**
   * Configure which rules engine variant to use for this session
   * (legacy, orchestrator adapter, or shadow) using the centralized
   * OrchestratorRolloutService.
   *
   * Phase A – Backend orchestrator-only path:
   * - The TurnEngineAdapter / shared orchestrator is now the only production
   *   backend turn-processing path whenever the global
   *   config.featureFlags.orchestrator.adapterEnabled flag is true.
   * - We no longer toggle the GameEngine's adapter on a per-session basis;
   *   per-session engineSelection is used solely for metrics and for deciding
   *   whether to run additional shadow comparisons (for example, Python or
   *   legacy pipelines) in diagnostics lanes.
   *
   * See docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md (Phase A – Backend orchestrator-only).
   */
  private configureEngineSelection(players: Player[]): void {
    // Prefer a human player for targeting; fall back to undefined.
    const primaryHuman = players.find((p) => p.type === 'human');
    const userId = primaryHuman?.id;

    const decision = orchestratorRollout.selectEngine(this.gameId, userId);

    // Adapter enablement is now controlled globally via
    // config.featureFlags.orchestrator.adapterEnabled (FSM is canonical).

    // Record a rollout session selection metric for observability.
    getMetricsService().recordOrchestratorSession(decision.engine, decision.reason);

    logger.info('Engine selection for game session', {
      gameId: this.gameId,
      engine: decision.engine,
      reason: decision.reason,
      targetedUserId: userId,
    });
  }

  private replayMove(move: PrismaMove): void {
    let from: Position | undefined;
    let to: Position | undefined;

    const rawPosition = move.position as unknown;
    if (typeof rawPosition === 'string') {
      try {
        const parsed = JSON.parse(rawPosition) as { from?: Position; to?: Position } | Position;
        if ('from' in parsed || 'to' in parsed) {
          const posObj = parsed as { from?: Position; to?: Position };
          from = posObj.from;
          to = posObj.to;
        } else {
          to = parsed as Position;
        }
      } catch (err) {
        logger.warn('Failed to parse persisted move.position string', {
          gameId: this.gameId,
          moveId: move.id,
          rawPosition,
          error: (err as Error).message,
        });
      }
    } else if (rawPosition && typeof rawPosition === 'object') {
      const parsed = rawPosition as { from?: Position; to?: Position } | Position;
      if ('from' in parsed || 'to' in parsed) {
        const posObj = parsed as { from?: Position; to?: Position };
        from = posObj.from;
        to = posObj.to;
      } else {
        to = parsed as Position;
      }
    }

    if (!to) {
      logger.warn('Skipping historical move with no destination', {
        gameId: this.gameId,
        moveId: move.id,
        rawPosition,
      });
      return;
    }

    // Map playerId (UUID) to playerNumber using the mapping built during initialization
    const playerNumber = this.playerIdToNumber.get(move.playerId);
    if (playerNumber === undefined) {
      logger.error('Failed to map playerId to playerNumber during move replay', {
        gameId: this.gameId,
        moveId: move.id,
        playerId: move.playerId,
        availablePlayerIds: Array.from(this.playerIdToNumber.keys()),
      });
      return;
    }

    const gameMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: move.moveType as MoveType,
      player: playerNumber,
      ...(from ? { from } : {}),
      to,
      thinkTime: 0,
    };

    try {
      this.gameEngine.makeMove(gameMove);
    } catch (err) {
      logger.error('Failed to replay historical move', {
        gameId: this.gameId,
        moveId: move.id,
        playerNumber,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  public getGameState(): GameState {
    return this.gameEngine.getGameState();
  }

  public getValidMoves(playerNumber: number): Move[] {
    return this.gameEngine.getValidMoves(playerNumber);
  }

  public getInteractionHandler(): WebSocketInteractionHandler {
    return this.wsHandler;
  }

  /**
   * Get the current session status snapshot
   */
  getSessionStatusSnapshot(): GameSessionStatus | null {
    return this.sessionStatus;
  }

  /**
   * Get the last AI request state (for testing/diagnostics)
   */
  getLastAIRequestStateForTesting(): AIRequestState {
    return this.aiRequestState;
  }

  /**
   * Get aggregated AI diagnostics snapshot (for testing/diagnostics)
   */
  getAIDiagnosticsSnapshotForTesting(): SessionAIDiagnostics {
    return { ...this.diagnosticsSnapshot };
  }

  /**
   * Recompute session status from current game state
   */
  recomputeSessionStatus(state?: GameState, result?: GameResult): void {
    const gameState = state ?? this.gameEngine?.getGameState();
    if (!gameState) {
      return;
    }

    const previousKind = this.sessionStatus?.kind;
    this.sessionStatus = deriveGameSessionStatus(gameState, result);
    const newKind = this.sessionStatus.kind;

    // Track status transitions for metrics
    if (previousKind !== newKind) {
      getMetricsService().recordGameSessionStatusTransition(previousKind ?? 'none', newKind);
      getMetricsService().updateGameSessionStatusCurrent(previousKind ?? null, newKind);

      logger.info('Game session status changed', {
        gameId: this.gameId,
        from: previousKind,
        to: newKind,
      });
    }
  }

  /**
   * Update aggregated diagnostics for a player
   *
   * NOTE: Shadow mode has been removed. FSM is now canonical.
   * pythonEvalFailures now used instead of pythonShadowErrors.
   */
  updateDiagnostics(playerNumber: number): void {
    // Get rules facade diagnostics
    const rulesDiag = this.rulesFacade?.getDiagnostics?.() ?? {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
    };

    // Get AI engine diagnostics for this player
    const aiDiag: AIDiagnostics = globalAIEngine.getDiagnostics(playerNumber) ?? {
      serviceFailureCount: 0,
      localFallbackCount: 0,
    };

    this.diagnosticsSnapshot = {
      rulesServiceFailureCount: rulesDiag.pythonEvalFailures,
      rulesShadowErrorCount: 0, // Shadow mode removed - FSM is canonical
      aiServiceFailureCount: aiDiag.serviceFailureCount,
      aiFallbackMoveCount: aiDiag.localFallbackCount,
      aiQualityMode: this.computeAIQualityMode(rulesDiag, aiDiag),
    };
  }

  private computeAIQualityMode(
    rulesDiag: { pythonEvalFailures: number },
    aiDiag: AIDiagnostics
  ): 'normal' | 'fallbackLocalAI' | 'rulesServiceDegraded' {
    // NOTE: Shadow mode removed - now checking pythonEvalFailures instead
    if (rulesDiag.pythonEvalFailures > 0) {
      return 'rulesServiceDegraded';
    }
    if (aiDiag.localFallbackCount > 0) {
      return 'fallbackLocalAI';
    }
    return 'normal';
  }

  /**
   * Map GameResult.reason to GameOutcome for record storage.
   */
  private mapGameResultToOutcome(gameResult: GameResult): GameOutcome {
    const reason = gameResult.reason;
    switch (reason) {
      case 'ring_elimination':
        return 'ring_elimination';
      case 'territory_control':
        return 'territory_control';
      case 'last_player_standing':
        return 'last_player_standing';
      case 'resignation':
        return 'resignation';
      case 'timeout':
        return 'timeout';
      case 'abandonment':
        return 'abandonment';
      case 'draw':
        return 'draw';
      case 'game_completed':
        // game_completed is a generic terminal reason; map to draw as fallback
        return 'draw';
      default:
        return 'last_player_standing';
    }
  }

  /**
   * Compute FinalScore from the final GameState.
   */
  private computeFinalScore(state: GameState): FinalScore {
    const ringsEliminated: Record<number, number> = {};
    const territorySpaces: Record<number, number> = {};
    const ringsRemaining: Record<number, number> = {};

    for (const player of state.players) {
      ringsEliminated[player.playerNumber] = player.eliminatedRings;
      territorySpaces[player.playerNumber] = player.territorySpaces;
      ringsRemaining[player.playerNumber] = player.ringsInHand;
    }

    return {
      ringsEliminated,
      territorySpaces,
      ringsRemaining,
    };
  }

  public async handlePlayerMove(
    socket: AuthenticatedSocket,
    moveData: PlayerMoveData
  ): Promise<void> {
    if (!socket.userId) throw new Error('Socket not authenticated');

    await this.handlePlayerMoveForUser(socket.userId, moveData);
  }

  /**
   * HTTP harness entry point for applying a move on behalf of a specific
   * authenticated userId. This mirrors the semantics of the WebSocket
   * player_move path but is transport-agnostic so that Express routes
   * can reuse the same domain pipeline.
   */
  public async handlePlayerMoveFromHttp(
    userId: string,
    moveData: PlayerMoveData
  ): Promise<RulesResult> {
    return this.handlePlayerMoveForUser(userId, moveData);
  }

  /**
   * Core move application pipeline shared by both WebSocket and HTTP
   * transports. This method performs:
   *
   * - Database/game-status validation
   * - GameState lookup and player vs spectator authorization
   * - Wire-level position -> engine Move conversion
   * - RulesBackendFacade.applyMove (including orchestrator/shadow handling)
   * - Persistence, broadcast, and AI turn chaining
   */
  private async handlePlayerMoveForUser(
    userId: string,
    moveData: PlayerMoveData
  ): Promise<RulesResult> {
    const prisma = getDatabaseClient();
    if (!prisma) throw new Error('Database not available');

    const game = await prisma.game.findUnique({ where: { id: this.gameId } });
    if (!game) throw new Error('Game not found');
    if (game.status !== 'active') throw new Error('Game is not active');

    const currentState = this.gameEngine.getGameState();
    const player = currentState.players.find((p) => p.id === userId);
    if (!player) {
      // Check if user is a spectator
      if (currentState.spectators.includes(userId)) {
        throw new Error('Spectators cannot make moves');
      }
      throw new Error('Current user is not a player in this game');
    }

    let from: Position | undefined;
    let to: Position | undefined;
    let placementCount: number | undefined;
    let placedOnStack: boolean | undefined;
    let captureTarget: Position | undefined;

    if (typeof moveData.position === 'string') {
      try {
        const parsed = JSON.parse(moveData.position);
        from = parsed.from as Position | undefined;
        to = (parsed.to as Position | undefined) ?? (parsed as Position);
        // Extract placement-specific fields for ring placement moves
        placementCount =
          typeof parsed.placementCount === 'number' ? parsed.placementCount : undefined;
        placedOnStack =
          typeof parsed.placedOnStack === 'boolean' ? parsed.placedOnStack : undefined;
        // Extract capture-specific fields for capture moves
        captureTarget = parsed.captureTarget as Position | undefined;
      } catch (err) {
        logger.warn('Failed to parse move.position payload', {
          gameId: this.gameId,
          rawPosition: moveData.position,
          error: (err as Error).message,
        });
        throw new Error('Invalid move position payload');
      }
    } else if (moveData.position && typeof moveData.position === 'object') {
      const parsed = moveData.position as
        | {
            from?: Position;
            to?: Position;
            placementCount?: number;
            placedOnStack?: boolean;
            captureTarget?: Position;
          }
        | Position;
      if ('from' in parsed || 'to' in parsed) {
        const posObj = parsed as {
          from?: Position;
          to?: Position;
          placementCount?: number;
          placedOnStack?: boolean;
          captureTarget?: Position;
        };
        from = posObj.from;
        to = (posObj.to as Position | undefined) ?? (posObj as unknown as Position | undefined);
        placementCount = posObj.placementCount;
        placedOnStack = posObj.placedOnStack;
        captureTarget = posObj.captureTarget;
      } else {
        to = parsed as Position;
      }
    }

    if (!to) throw new Error('Move destination is required');

    const engineMove = {
      player: player.playerNumber,
      type: moveData.moveType as Move['type'],
      from,
      to,
      placementCount,
      placedOnStack,
      captureTarget,
      thinkTime: 0,
    } as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;

    const result = await this.rulesFacade.applyMove(engineMove);
    if (!result.success) {
      logger.warn('Engine rejected move', {
        gameId: this.gameId,
        userId,
        reason: result.error,
      });
      throw new Error(result.error || 'Invalid move');
    }

    await this.persistMove(userId, result);
    await this.broadcastUpdate(result);
    await this.maybePerformAITurn();

    return result;
  }

  public async handlePlayerMoveById(socket: AuthenticatedSocket, moveId: string): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) throw new Error('Database not available');
    if (!socket.userId) throw new Error('Socket not authenticated');

    const game = await prisma.game.findUnique({ where: { id: this.gameId } });
    if (!game) throw new Error('Game not found');
    if (game.status !== 'active') throw new Error('Game is not active');

    const currentState = this.gameEngine.getGameState();
    const userId = socket.userId;
    const player = currentState.players.find((p) => p.id === userId);
    if (!player) {
      // Check if user is a spectator
      if (currentState.spectators.includes(userId)) {
        throw new Error('Spectators cannot make moves');
      }
      throw new Error('Current user is not a player in this game');
    }

    const result = await this.rulesFacade.applyMoveById(player.playerNumber, moveId);
    if (!result.success) {
      logger.warn('Engine rejected move by id', {
        gameId: this.gameId,
        userId,
        moveId,
        reason: result.error,
      });
      throw new Error(result.error || 'Invalid move selection');
    }

    await this.persistMove(userId, result);
    await this.broadcastUpdate(result);
    await this.maybePerformAITurn();
  }

  /**
   * Handle a clean resignation initiated by a specific userId (for example
   * via the HTTP /games/:gameId/leave route when the game is active).
   */
  public async handlePlayerResignationByUserId(userId: string): Promise<GameResult | null> {
    const currentState = this.gameEngine.getGameState();
    if (currentState.gameStatus !== 'active') {
      return null;
    }

    const player = currentState.players.find((p) => p.id === userId && p.type === 'human');
    if (!player) {
      return null;
    }

    const engineResult = this.gameEngine.resignPlayer(player.playerNumber);
    const updatedState = this.gameEngine.getGameState();

    await this.finishGameWithResult(updatedState, engineResult.gameResult);
    await this.broadcastUpdate(engineResult);

    return engineResult.gameResult;
  }

  /**
   * Handle game-level abandonment after a player's reconnect window has
   * expired. When awardWinToOpponent is true, a remaining opponent is
   * credited as the winner via GameEngine.abandonPlayer; otherwise the
   * game is marked as abandoned without a specific winner.
   */
  public async handleAbandonmentForDisconnectedPlayer(
    playerNumber: number,
    awardWinToOpponent: boolean
  ): Promise<GameResult | null> {
    const currentState = this.gameEngine.getGameState();
    if (currentState.gameStatus !== 'active') {
      return null;
    }

    const engineResult = awardWinToOpponent
      ? this.gameEngine.abandonPlayer(playerNumber)
      : this.gameEngine.abandonGameAsDraw();

    const updatedState = this.gameEngine.getGameState();

    await this.finishGameWithResult(updatedState, engineResult.gameResult);
    await this.broadcastUpdate(engineResult);

    return engineResult.gameResult;
  }

  private async persistMove(playerId: string, result: RulesResult): Promise<void> {
    const updatedState = this.gameEngine.getGameState();
    const lastMove = updatedState.moveHistory[updatedState.moveHistory.length - 1];

    if (lastMove) {
      // Use GamePersistenceService for async, non-blocking move saving
      GamePersistenceService.saveMove({
        gameId: this.gameId,
        playerId,
        moveNumber: lastMove.moveNumber,
        move: lastMove,
      });
    }

    if (result.gameResult) {
      await this.finishGameWithResult(updatedState, result.gameResult);
    }
  }

  /**
   * Common helper for finishing a game given the final GameState and
   * canonical GameResult produced by the engine. This is used both for
   * move-driven terminations (via persistMove) and host-driven flows
   * such as resignations and disconnect-induced abandonment.
   */
  private async finishGameWithResult(state: GameState, gameResult: GameResult): Promise<void> {
    const winnerPlayerNumber = gameResult.winner;
    let winnerId: string | null = null;

    if (winnerPlayerNumber !== undefined) {
      const winnerPlayer = state.players.find(
        (p) => p.playerNumber === winnerPlayerNumber && p.type === 'human'
      );
      winnerId = winnerPlayer?.id ?? null;
    }

    await GamePersistenceService.finishGame(this.gameId, winnerId, state, gameResult);

    // Best-effort persistence of a canonical GameRecord row for this
    // completed game. This populates finalState/finalScore/outcome and
    // recordMetadata so that GameRecordRepository and JSONL exporters
    // have a stable schema to work with.
    try {
      const outcome = this.mapGameResultToOutcome(gameResult);
      const finalScore = this.computeFinalScore(state);
      await gameRecordRepository.saveGameRecord(this.gameId, state, outcome, finalScore, {
        source: 'online_game',
        tags: state.isRated ? ['rated'] : ['unrated'],
      });
    } catch (err) {
      // Non-fatal: GameRecord storage failures should not affect game completion.
      logger.warn('Failed to save canonical GameRecord', {
        gameId: this.gameId,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  private async broadcastUpdate(
    result: RulesResult,
    decisionMeta?: DecisionAutoResolvedMeta | null
  ): Promise<void> {
    const updatedState = this.gameEngine.getGameState();

    if (result.gameResult) {
      this.io.to(this.gameId).emit('game_over', {
        type: 'game_over',
        data: {
          gameId: this.gameId,
          gameState: updatedState,
          gameResult: result.gameResult,
        },
        timestamp: new Date().toISOString(),
      });

      // Update session status for game completion
      this.recomputeSessionStatus(updatedState, result.gameResult);

      // Game is over – clear any pending decision-phase timeout.
      this.resetDecisionPhaseTimeout();

      // Game is over – cancel any in-flight player choices so UI and
      // decision Promises don't linger past the terminal state.
      this.wsHandler?.cancelAllChoices();
    } else {
      // Broadcast state to all connected clients in the room
      // For active players, we include their valid moves
      // For spectators, validMoves is empty
      const room = this.io.sockets.adapter.rooms.get(this.gameId);
      if (room) {
        for (const socketId of room) {
          const socket = this.io.sockets.sockets.get(socketId) as AuthenticatedSocket | undefined;
          if (!socket) continue;

          const isPlayer = updatedState.players.some((p) => p.id === socket.userId);
          // Only send valid moves to the active player
          const isActivePlayer =
            isPlayer &&
            updatedState.players.find((p) => p.id === socket.userId)?.playerNumber ===
              updatedState.currentPlayer;
          const validMoves = isActivePlayer
            ? this.gameEngine.getValidMoves(updatedState.currentPlayer)
            : [];

          // Serialize board Maps to plain objects for JSON transport while keeping
          // all other GameState fields intact. The client's hydrateBoardState()
          // reconstructs Maps from the plain objects.
          const transportState = {
            ...updatedState,
            board: serializeBoardState(updatedState.board),
          };

          const payload = {
            type: 'game_update' as const,
            data: {
              gameId: this.gameId,
              gameState: transportState as unknown as GameState,
              validMoves,
              ...(decisionMeta && {
                meta: {
                  diffSummary: {
                    decisionAutoResolved: decisionMeta,
                  },
                },
              }),
            },
            timestamp: new Date().toISOString(),
          };

          socket.emit('game_state', payload);
        }
      }

      // Update session status
      this.recomputeSessionStatus(updatedState);

      // (Re)schedule decision-phase timeout based on the new state.
      this.scheduleDecisionPhaseTimeout(updatedState);

      // Best-effort analysis-mode position evaluation stream. This is gated
      // behind a feature flag so that operators can disable it entirely if
      // needed without affecting core gameplay.
      if (this.isAnalysisModeEnabled()) {
        void this.evaluateAndBroadcastPosition(updatedState);
      }
    }
  }

  /**
   * Check whether AI analysis mode (position evaluation streaming) is enabled
   * for this process.
   */
  private isAnalysisModeEnabled(): boolean {
    return !!config.featureFlags?.analysisMode?.enabled;
  }

  /**
   * Call the Python AI service to evaluate the current position for all
   * players and broadcast the result as a best-effort WebSocket event.
   *
   * Failures are logged but intentionally do not affect gameplay.
   */
  private async evaluateAndBroadcastPosition(state: GameState): Promise<void> {
    try {
      const aiClient = getAIServiceClient();
      const response = await aiClient.evaluatePositionMulti(state);

      const payload: PositionEvaluationPayload = {
        type: 'position_evaluation',
        data: {
          gameId: this.gameId,
          moveNumber: response.move_number,
          boardType: (response.board_type as GameState['boardType']) ?? state.boardType,
          perPlayer: response.per_player,
          engineProfile: response.engine_profile,
          evaluationScale: response.evaluation_scale,
        },
        timestamp: response.generated_at,
      };

      this.io.to(this.gameId).emit('position_evaluation', payload);
    } catch (err) {
      logger.warn('Failed to emit position evaluation', {
        gameId: this.gameId,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  /**
   * Create a deterministic RNG for backend local AI decisions based on the
   * canonical GameState.rngSeed. This mirrors the seeding used by AIEngine so
   * that for a fixed rngSeed and game configuration, fallback and decision
   * phases are reproducible across runs.
   */
  private createLocalAIRng(state: GameState, playerNumber: number): LocalAIRng {
    const baseSeed = typeof state.rngSeed === 'number' ? state.rngSeed : 0;
    const mixed = (baseSeed ^ (playerNumber * 0x9e3779b1)) >>> 0;
    const rng = new SeededRNG(mixed);
    return () => rng.next();
  }

  /**
   * Perform AI turn with explicit state machine tracking and timeout handling.
   * This is the Phase 7 hardened implementation.
   */
  public async maybePerformAITurn(): Promise<void> {
    try {
      const state = this.gameEngine.getGameState();
      if (state.gameStatus !== 'active') return;

      const currentPlayerNumber = state.currentPlayer;
      const currentPlayer = state.players.find((p) => p.playerNumber === currentPlayerNumber);

      if (!currentPlayer || currentPlayer.type !== 'ai') return;

      const localRng = this.createLocalAIRng(state, currentPlayerNumber);

      // Cancel any in-flight request from previous turn
      this.cancelInFlightAIRequest('manual');

      // Start new AI request lifecycle
      this.aiRequestState = markQueued(Date.now(), this.aiRequestTimeoutMs);

      let aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      if (!aiConfig) {
        const difficulty = currentPlayer.aiDifficulty ?? 5;
        globalAIEngine.createAI(currentPlayerNumber, difficulty);
        aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      }

      logger.info('Starting AI turn', {
        gameId: this.gameId,
        playerNumber: currentPlayerNumber,
        phase: state.currentPhase,
      });

      // Create abort controller for this request
      this.aiAbortController = new AbortController();

      // Transition to in_flight
      this.aiRequestState = markInFlight(this.aiRequestState, Date.now(), this.aiRequestTimeoutMs);

      let result: RulesResult;
      let appliedMoveType: Move['type'] | undefined;

      try {
        if (
          state.currentPhase === 'line_processing' ||
          state.currentPhase === 'territory_processing'
        ) {
          // Handle decision phases
          const allCandidates = this.gameEngine.getValidMoves(currentPlayerNumber);
          const decisionCandidates = allCandidates.filter((m) => {
            if (state.currentPhase === 'line_processing') {
              return m.type === 'process_line' || m.type === 'choose_line_option';
            }
            return m.type === 'choose_territory_option' || m.type === 'eliminate_rings_from_stack';
          });

          if (decisionCandidates.length === 0) {
            this.aiAbortController = null;
            return;
          }

          const selected = globalAIEngine.chooseLocalMoveFromCandidates(
            currentPlayerNumber,
            state,
            decisionCandidates,
            localRng
          );

          if (!selected) {
            this.aiAbortController = null;
            return;
          }

          const {
            id: _id,
            timestamp: _timestamp,
            moveNumber: _moveNumber,
            ...rest
          } = selected as Move;
          const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
          appliedMoveType = engineMove.type;

          result = await this.rulesFacade.applyMove(engineMove);
        } else {
          // Get AI move with timeout
          const aiMove = await this.getAIMoveWithTimeout(
            currentPlayerNumber,
            state,
            this.aiRequestTimeoutMs,
            {
              token: this.sessionCancellationSource.token,
            }
          );

          // Check if request was canceled during execution
          if (this.aiAbortController?.signal.aborted) {
            return;
          }

          if (!aiMove) {
            await this.handleNoMoveFromService(currentPlayerNumber, state);
            return;
          }

          const { id: _id2, timestamp: _timestamp2, moveNumber: _moveNumber2, ...rest } = aiMove;
          const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
          appliedMoveType = engineMove.type;

          result = await this.rulesFacade.applyMove(engineMove);
        }

        if (!result.success) {
          logger.warn('Engine rejected AI move', {
            gameId: this.gameId,
            playerNumber: currentPlayerNumber,
            reason: result.error,
          });
          await this.handleServiceMoveRejected(currentPlayerNumber, state, result.error);
          return;
        }

        // Mark completed
        this.aiRequestState = markCompleted(this.aiRequestState);
        getMetricsService().recordAITurnRequestTerminal('completed');

        // Persist AI move
        await this.persistAIMove(currentPlayerNumber, appliedMoveType, result);

        // Update diagnostics
        this.updateDiagnostics(currentPlayerNumber);

        // Broadcast update
        await this.broadcastUpdate(result);

        logger.info('AI move processed and applied', {
          gameId: this.gameId,
          playerNumber: currentPlayerNumber,
          moveType: appliedMoveType,
          aiDiagnostics: this.diagnosticsSnapshot,
        });

        // Recursively check for next AI turn
        await this.maybePerformAITurn();
      } catch (error) {
        // Check if this was a timeout
        if (isDeadlineExceeded(this.aiRequestState)) {
          this.aiRequestState = markTimedOut(this.aiRequestState);
          getMetricsService().recordAITurnRequestTerminal('timed_out');
          getMetricsService().recordAIFallback('timeout');

          // After markTimedOut, aiRequestState.kind is 'timed_out' which has durationMs
          const timedOutState = this.aiRequestState as { kind: 'timed_out'; durationMs: number };
          logger.warn('AI request timed out', {
            gameId: this.gameId,
            playerNumber: currentPlayerNumber,
            durationMs: timedOutState.durationMs,
          });

          // Try local fallback after timeout
          await this.handleNoMoveFromService(currentPlayerNumber, state);
        } else {
          // Some other error - extract aiErrorType if available from error object
          const errorObj = error as { aiErrorType?: string } | null;
          const errorType = errorObj?.aiErrorType ?? 'unknown';
          this.aiRequestState = markFailed(
            'AI_SERVICE_ERROR',
            errorType,
            this.aiRequestState,
            Date.now()
          );

          getMetricsService().recordAITurnRequestTerminal('failed', 'AI_SERVICE_ERROR', errorType);

          logger.error('AI turn failed', {
            gameId: this.gameId,
            playerNumber: currentPlayerNumber,
            error: error instanceof Error ? error.message : String(error),
          });

          // Try local fallback after error
          await this.handleNoMoveFromService(currentPlayerNumber, state);
        }
      } finally {
        this.aiAbortController = null;
      }
    } catch (error) {
      logger.error('Error during AI turn', {
        gameId: this.gameId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Get AI move with timeout.
   *
   * This wraps the AI engine call in the shared runWithTimeout helper so that
   * the timeout semantics are centralized alongside other Tier 3 async flows.
   * The state machine still owns the authoritative view of deadlines via
   * aiRequestState.deadlineAt; on timeout we surface a synthetic error and
   * let the caller map it to the explicit `timed_out` terminal state using
   * the existing isDeadlineExceeded/markTimedOut logic.
   */
  private async getAIMoveWithTimeout(
    playerNumber: number,
    state: GameState,
    timeoutMs: number,
    options?: { token?: CancellationToken }
  ): Promise<Move | null> {
    const result = await runWithTimeout(
      () =>
        globalAIEngine.getAIMove(
          playerNumber,
          state,
          undefined,
          options?.token ? { token: options.token } : undefined
        ),
      {
        timeoutMs,
        ...(options?.token && { token: options.token }),
      }
    );

    if (result.kind === 'ok') {
      return result.value ?? null;
    }

    if (result.kind === 'timeout') {
      const error: Error & { isTimeout?: boolean } = new Error('AI request timeout');
      // Mark as a timeout-specific error so future callers that choose to
      // inspect the error object (for example metrics) can distinguish it
      // from generic failures, even though GameSession currently relies on
      // the state-machine deadline for timeout detection.
      error.isTimeout = true;
      throw error;
    }

    if (result.kind === 'canceled') {
      const error: Error & { cancellationReason?: unknown } = new Error('AI request canceled');
      error.cancellationReason = result.cancellationReason;
      throw error;
    }

    // Exhaustiveness guard – TimedOperationOutcome is a closed union. If we
    // land here, the helper's API has changed and this method should be
    // updated accordingly.
    throw new Error('Unhandled TimedOperationResult outcome in getAIMoveWithTimeout');
  }

  /**
   * Handle when service returns a move that's rejected by rules engine
   */
  private async handleServiceMoveRejected(
    playerNumber: number,
    state: GameState,
    error: string | undefined
  ): Promise<void> {
    logger.warn('Service move rejected by rules engine, trying fallback', {
      gameId: this.gameId,
      playerNumber,
      error,
    });

    // Transition to fallback_local
    this.aiRequestState = markFallbackLocal(this.aiRequestState);
    getMetricsService().recordAIFallback('move_rejected');

    // Try local fallback using a deterministic RNG derived from the
    // canonical GameState.rngSeed so fallback behaviour is reproducible.
    const fallbackRng = this.createLocalAIRng(state, playerNumber);
    const fallbackMove = globalAIEngine.getLocalFallbackMove(playerNumber, state, fallbackRng);

    if (fallbackMove) {
      const result = await this.rulesFacade.applyMove(fallbackMove);

      if (result.success) {
        this.aiRequestState = markCompleted(this.aiRequestState);
        getMetricsService().recordAITurnRequestTerminal('completed');

        await this.persistAIMove(playerNumber, fallbackMove.type, result);
        await this.broadcastUpdate(result);
        await this.maybePerformAITurn();
      } else {
        // Both service and fallback failed
        await this.handleAIFatalFailure(playerNumber, 'Fallback move also rejected');
      }
    } else {
      await this.handleAIFatalFailure(playerNumber, 'No fallback move available');
    }
  }

  /**
   * Handle when service returns no move
   */
  private async handleNoMoveFromService(playerNumber: number, state: GameState): Promise<void> {
    logger.warn('No move from AI service, trying fallback', {
      gameId: this.gameId,
      playerNumber,
    });

    // Transition to fallback_local
    this.aiRequestState = markFallbackLocal(this.aiRequestState);
    getMetricsService().recordAIFallback('no_move');

    // Try local fallback using a deterministic RNG derived from the
    // canonical GameState.rngSeed so fallback behaviour is reproducible.
    const fallbackRng = this.createLocalAIRng(state, playerNumber);
    const fallbackMove = globalAIEngine.getLocalFallbackMove(playerNumber, state, fallbackRng);

    if (fallbackMove) {
      const result = await this.rulesFacade.applyMove(fallbackMove);

      if (result.success) {
        this.aiRequestState = markCompleted(this.aiRequestState);
        getMetricsService().recordAITurnRequestTerminal('completed');

        await this.persistAIMove(playerNumber, fallbackMove.type, result);
        await this.broadcastUpdate(result);
        await this.maybePerformAITurn();
      } else {
        await this.handleAIFatalFailure(playerNumber, 'Fallback move rejected');
      }
    } else {
      await this.handleAIFatalFailure(playerNumber, 'No fallback move available');
    }
  }

  /**
   * Handle fatal AI failure (both service and fallback failed)
   *
   * As a last resort, tries to apply any valid move to prevent the game from stalling.
   */
  private async handleAIFatalFailure(playerNumber: number, reason: string): Promise<void> {
    this.aiRequestState = markFailed(
      'AI_SERVICE_OVERLOADED',
      'both_service_and_fallback_failed',
      this.aiRequestState,
      Date.now()
    );

    getMetricsService().recordAITurnRequestTerminal(
      'failed',
      'AI_SERVICE_OVERLOADED',
      'both_service_and_fallback_failed'
    );

    logger.error('AI fatal failure - both service and fallback failed', {
      gameId: this.gameId,
      playerNumber,
      reason,
    });

    // Update diagnostics
    this.updateDiagnostics(playerNumber);

    // Last resort recovery: pick the first valid move and apply it
    // This prevents the game from stalling indefinitely when AI fails
    try {
      const validMoves = this.gameEngine.getValidMoves(playerNumber);
      if (validMoves.length > 0) {
        const emergencyMove = validMoves[0];
        logger.warn('AI fatal failure recovery: applying first valid move', {
          gameId: this.gameId,
          playerNumber,
          moveType: emergencyMove.type,
        });

        const result = await this.rulesFacade.applyMove(emergencyMove);
        if (result.success) {
          await this.persistAIMove(playerNumber, emergencyMove.type, result);
          await this.broadcastUpdate(result);
          await this.maybePerformAITurn();
        } else {
          logger.error('AI fatal failure recovery move rejected', {
            gameId: this.gameId,
            playerNumber,
            error: result.error,
          });
        }
      } else {
        logger.error('AI fatal failure: no valid moves available for recovery', {
          gameId: this.gameId,
          playerNumber,
        });
      }
    } catch (recoveryError) {
      logger.error('AI fatal failure recovery failed', {
        gameId: this.gameId,
        playerNumber,
        error: recoveryError instanceof Error ? recoveryError.message : String(recoveryError),
      });
    }
  }

  /**
   /**
    * Persist AI move to database using GamePersistenceService
    */
  private async persistAIMove(
    playerNumber: number,
    _moveType: Move['type'] | undefined,
    result: RulesResult
  ): Promise<void> {
    try {
      const aiUser = await getOrCreateAIUser();
      const updatedState = this.gameEngine.getGameState();
      const lastMove = updatedState.moveHistory[updatedState.moveHistory.length - 1];

      if (lastMove) {
        // Use GamePersistenceService for async, non-blocking move saving
        GamePersistenceService.saveMove({
          gameId: this.gameId,
          playerId: aiUser.id,
          moveNumber: lastMove.moveNumber,
          move: lastMove,
        });
      }

      if (result.gameResult) {
        // Reuse the shared completion helper so that AI-terminated games persist a
        // canonical GameRecord row with finalState/finalScore/outcome and
        // recordMetadata, matching the behaviour of player-driven terminations.
        await this.finishGameWithResult(updatedState, result.gameResult);
      }
    } catch (err) {
      logger.error('Failed to persist AI move', {
        gameId: this.gameId,
        playerNumber,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
  /**
   * Cancel any in-flight AI request
   */
  cancelInFlightAIRequest(reason: AIRequestCancelReason): void {
    if (isCancelable(this.aiRequestState)) {
      this.aiRequestState = markCanceled(reason, this.aiRequestState);

      getMetricsService().recordAITurnRequestTerminal('canceled', 'none', reason);

      if (this.aiAbortController) {
        this.aiAbortController.abort();
        this.aiAbortController = null;
      }

      logger.info('AI request canceled', {
        gameId: this.gameId,
        reason,
      });
    }
  }

  /**
   * Terminate the session and cancel any pending operations
   */
  terminate(reason: AIRequestCancelReason = 'session_cleanup'): void {
    // Cooperatively cancel any session-scoped async work (e.g. AI turns)
    // before updating local state machines and timers.
    this.sessionCancellationSource.cancel(reason);

    // Cancel any in-flight AI request
    this.cancelInFlightAIRequest(reason);

    // Clear any active decision-phase timeout timers so that no further
    // auto-resolution or warning events fire after session termination.
    this.resetDecisionPhaseTimeout();

    // Update session status to reflect termination
    const state = this.getGameState();
    if (state.gameStatus === 'active') {
      // Mark as abnormal termination
      getMetricsService().recordGameSessionStatusTransition(
        this.sessionStatus?.kind ?? 'active_turn',
        'abandoned'
      );
      getMetricsService().recordAbnormalTermination(reason);
    }

    logger.info('GameSession terminated', {
      gameId: this.gameId,
      reason,
    });
  }

  // ===================
  // Decision Phase Timeout API
  // ===================

  /**
   * Get remaining time for current decision phase timeout.
   * Returns null if no timeout is active.
   */
  getDecisionPhaseRemainingMs(): number | null {
    if (this.decisionTimeoutDeadlineMs == null) {
      return null;
    }
    const remaining = this.decisionTimeoutDeadlineMs - Date.now();
    return remaining > 0 ? remaining : 0;
  }

  /**
   * Reset/clear the current decision phase timeout.
   */
  resetDecisionPhaseTimeout(): void {
    if (this.decisionTimeoutWarningHandle) {
      clearTimeout(this.decisionTimeoutWarningHandle);
      this.decisionTimeoutWarningHandle = null;
    }
    if (this.decisionTimeoutHandle) {
      clearTimeout(this.decisionTimeoutHandle);
      this.decisionTimeoutHandle = null;
    }

    this.decisionTimeoutDeadlineMs = null;
    this.decisionTimeoutPhase = null;
    this.decisionTimeoutPlayer = null;
    this.decisionTimeoutChoiceType = null;
    this.decisionTimeoutChoiceKind = null;
  }

  /**
   * Internal helper to map a GamePhase to the narrower timeout phase union
   * used by DecisionPhaseTimeoutWarningPayload / DecisionPhaseTimedOutPayload.
   */
  private mapPhaseToTimeoutPhase(
    phase: GamePhase
  ): DecisionPhaseTimeoutWarningPayload['data']['phase'] | null {
    if (
      phase === 'line_processing' ||
      phase === 'territory_processing' ||
      phase === 'chain_capture'
    ) {
      return phase;
    }
    return null;
  }

  /**
   * Inspect the latest GameState and (re)schedule decision-phase timeout
   * timers when the active player is a human and the phase exposes
   * canonical decision moves.
   */
  private scheduleDecisionPhaseTimeout(state: GameState): void {
    // Always clear any previous timers first.
    this.resetDecisionPhaseTimeout();

    if (state.gameStatus !== 'active') {
      return;
    }

    const phase = state.currentPhase;
    const timeoutPhase = this.mapPhaseToTimeoutPhase(phase);
    if (!timeoutPhase) {
      return;
    }

    const currentPlayerNumber = state.currentPlayer;
    const currentPlayer = state.players.find((p) => p.playerNumber === currentPlayerNumber);
    if (!currentPlayer || currentPlayer.type !== 'human') {
      return;
    }

    const allMoves = this.gameEngine.getValidMoves(currentPlayerNumber);

    const classification = this.classifyDecisionSurface(phase, allMoves);
    if (!classification) {
      return;
    }

    const { choiceType, choiceKind, candidateTypes } = classification;
    const decisionCandidates = allMoves.filter((m) => candidateTypes.includes(m.type));
    if (decisionCandidates.length === 0) {
      return;
    }

    const now = Date.now();
    // Use fixture timeout override if available (for E2E testing), otherwise use global config
    const timeoutMs =
      this.fixtureMetadata?.shortTimeoutMs ?? config.decisionPhaseTimeouts.defaultTimeoutMs;
    const warningBeforeMs =
      this.fixtureMetadata?.shortWarningBeforeMs ??
      config.decisionPhaseTimeouts.warningBeforeTimeoutMs;

    this.decisionTimeoutDeadlineMs = now + timeoutMs;
    this.decisionTimeoutPhase = phase;
    this.decisionTimeoutPlayer = currentPlayerNumber;
    this.decisionTimeoutChoiceType = choiceType;
    this.decisionTimeoutChoiceKind = choiceKind;

    const warningDelay = timeoutMs - warningBeforeMs;

    if (warningDelay > 0) {
      this.decisionTimeoutWarningHandle = setTimeout(() => {
        this.emitDecisionPhaseTimeoutWarning();
      }, warningDelay);
    }

    this.decisionTimeoutHandle = setTimeout(() => {
      void this.handleDecisionPhaseTimedOut();
    }, timeoutMs);

    logger.info('Scheduled decision phase timeout', {
      gameId: this.gameId,
      playerNumber: currentPlayerNumber,
      phase,
      timeoutMs,
      warningBeforeMs,
      fixtureOverride: !!this.fixtureMetadata?.shortTimeoutMs,
    });
  }

  /**
   * Derive the logical PlayerChoiceType / DecisionChoiceKind and candidate
   * Move types for the current decision surface.
   */
  private classifyDecisionSurface(
    phase: GamePhase,
    moves: Move[]
  ): {
    choiceType: PlayerChoiceType;
    choiceKind: DecisionAutoResolvedMeta['choiceKind'];
    candidateTypes: MoveType[];
  } | null {
    if (phase === 'line_processing') {
      if (moves.some((m) => m.type === 'process_line')) {
        return {
          choiceType: 'line_order',
          choiceKind: 'line_order',
          candidateTypes: ['process_line'],
        };
      }
      if (moves.some((m) => m.type === 'choose_line_option')) {
        return {
          choiceType: 'line_reward_option',
          choiceKind: 'line_reward',
          candidateTypes: ['choose_line_option'],
        };
      }
      return null;
    }

    if (phase === 'territory_processing') {
      if (moves.some((m) => m.type === 'choose_territory_option')) {
        return {
          choiceType: 'region_order',
          choiceKind: 'territory_region_order',
          candidateTypes: ['choose_territory_option', 'skip_territory_processing'],
        };
      }
      if (moves.some((m) => m.type === 'eliminate_rings_from_stack')) {
        return {
          choiceType: 'ring_elimination',
          choiceKind: 'ring_elimination',
          candidateTypes: ['eliminate_rings_from_stack'],
        };
      }
      return null;
    }

    if (phase === 'chain_capture') {
      if (moves.some((m) => m.type === 'continue_capture_segment')) {
        return {
          choiceType: 'capture_direction',
          choiceKind: 'capture_direction',
          candidateTypes: ['continue_capture_segment'],
        };
      }
      return null;
    }

    return null;
  }

  /**
   * Emit a warning event shortly before the decision timeout elapses.
   */
  private emitDecisionPhaseTimeoutWarning(): void {
    if (
      this.decisionTimeoutDeadlineMs == null ||
      this.decisionTimeoutPlayer == null ||
      !this.decisionTimeoutPhase
    ) {
      return;
    }

    const timeoutPhase = this.mapPhaseToTimeoutPhase(this.decisionTimeoutPhase);
    if (!timeoutPhase) {
      return;
    }

    const remainingMs = Math.max(0, this.decisionTimeoutDeadlineMs - Date.now());

    const payload: DecisionPhaseTimeoutWarningPayload = {
      type: 'decision_phase_timeout_warning',
      data: {
        gameId: this.gameId,
        playerNumber: this.decisionTimeoutPlayer,
        phase: timeoutPhase,
        remainingMs,
      },
      timestamp: new Date().toISOString(),
    };

    this.io.to(this.gameId).emit('decision_phase_timeout_warning', payload);

    logger.info('Emitted decision phase timeout warning', {
      gameId: this.gameId,
      playerNumber: this.decisionTimeoutPlayer,
      phase: timeoutPhase,
      remainingMs,
    });
  }

  /**
   * Handle an expired decision-phase timeout by auto-selecting a canonical
   * Move for the active human player, applying it, persisting it, emitting
   * timeout events, and broadcasting an update with decisionAutoResolved
   * metadata attached.
   *
   * Selection MUST be deterministic across hosts. To avoid depending on
   * incidental getValidMoves() ordering, we derive a stable sort key from
   * geometric fields on the candidate Moves (lines, regions, stacks, or
   * capture geometry) and then pick the lexicographically-smallest
   * candidate. This keeps auto-resolution behaviour reproducible and
   * aligned with orchestrator traces (see P18.3-1 §3.1.2 / §6.4).
   */
  private async handleDecisionPhaseTimedOut(): Promise<void> {
    const phaseSnapshot = this.decisionTimeoutPhase;
    const playerSnapshot = this.decisionTimeoutPlayer;
    const choiceTypeSnapshot = this.decisionTimeoutChoiceType;
    const choiceKindSnapshot = this.decisionTimeoutChoiceKind;

    // Clear timers up front to avoid duplicate handling; scheduling for the
    // next decision surface is driven by broadcastUpdate.
    this.resetDecisionPhaseTimeout();

    if (
      phaseSnapshot == null ||
      playerSnapshot == null ||
      !choiceTypeSnapshot ||
      !choiceKindSnapshot
    ) {
      return;
    }

    // P0 FIX: Acquire lock before reading/modifying game state to prevent
    // race conditions with concurrent player moves (RingRift-2026-01-11).
    await this.withLock(async () => {
      await this.handleDecisionPhaseTimedOutLocked(
        phaseSnapshot,
        playerSnapshot,
        choiceTypeSnapshot,
        choiceKindSnapshot
      );
    });
  }

  /**
   * Lock-protected implementation of decision phase timeout handling.
   * Called by handleDecisionPhaseTimedOut after acquiring the game lock.
   */
  private async handleDecisionPhaseTimedOutLocked(
    phaseSnapshot: GamePhase,
    playerSnapshot: number,
    choiceTypeSnapshot: PlayerChoiceType,
    choiceKindSnapshot: DecisionAutoResolvedMeta['choiceKind']
  ): Promise<void> {
    const state = this.gameEngine.getGameState();
    if (state.gameStatus !== 'active') {
      return;
    }

    if (state.currentPlayer !== playerSnapshot) {
      return;
    }

    if (state.currentPhase !== phaseSnapshot) {
      return;
    }

    const timeoutPhase = this.mapPhaseToTimeoutPhase(state.currentPhase);
    if (!timeoutPhase) {
      return;
    }

    const currentPlayer = state.players.find((p) => p.playerNumber === playerSnapshot);
    if (!currentPlayer || currentPlayer.type !== 'human') {
      return;
    }

    const allMoves = this.gameEngine.getValidMoves(playerSnapshot);
    const classification = this.classifyDecisionSurface(state.currentPhase, allMoves);
    if (!classification) {
      return;
    }

    const { candidateTypes } = classification;
    const decisionCandidates = allMoves.filter((m) => candidateTypes.includes(m.type));
    if (decisionCandidates.length === 0) {
      return;
    }

    /**
     * Build a stable, geometry-driven sort key for a decision candidate.
     * This relies only on canonical Move fields and is therefore
     * deterministic for a given GameState, independent of host/order.
     */
    const positionKey = (pos: Position | undefined | null): string => {
      if (!pos) {
        return '~';
      }
      // Position has x, y required; z is optional
      const x = typeof pos.x === 'number' ? pos.x : 0;
      const y = typeof pos.y === 'number' ? pos.y : 0;
      const z = typeof pos.z === 'number' ? pos.z : 0;
      return `${x},${y},${z}`;
    };

    const positionsKey = (positions: Position[] | undefined | null): string => {
      if (!positions || positions.length === 0) {
        return '~';
      }
      return positions.map((p) => positionKey(p)).join('|');
    };

    const sortKeyForDecisionMove = (move: Move): string => {
      switch (move.type) {
        case 'process_line':
        case 'choose_line_option': {
          // Move.formedLines is optional LineInfo[] on the Move type
          const formedLines = move.formedLines as LineInfo[] | undefined;
          const primaryLine = formedLines && formedLines.length > 0 ? formedLines[0] : null;
          const lineKey = primaryLine ? positionsKey(primaryLine.positions) : '~';
          return `line:${lineKey}`;
        }
        case 'choose_territory_option': {
          // Move.disconnectedRegions is optional Territory[] on the Move type
          const regions = move.disconnectedRegions as Territory[] | undefined;
          const primaryRegion = regions && regions.length > 0 ? regions[0] : null;
          const regionKey = primaryRegion ? positionsKey(primaryRegion.spaces) : '~';
          return `territory:${regionKey}`;
        }
        case 'eliminate_rings_from_stack': {
          // Move.eliminationFromStack is optional on the Move type
          const eliminationFromStack = move.eliminationFromStack;
          const basePos = move.to || eliminationFromStack?.position;
          const stackKey = positionKey(basePos ?? null);
          return `elim:${stackKey}`;
        }
        case 'continue_capture_segment': {
          const fromKey = positionKey(move.from ?? null);
          // Move.captureTarget is optional Position on the Move type
          const captureTarget = move.captureTarget;
          const targetKey = positionKey(captureTarget ?? null);
          const toKey = positionKey(move.to ?? null);
          return `chain:${fromKey}|${targetKey}|${toKey}`;
        }
        default:
          // Fallback for any future decision types: rely on type + id only.
          return `other:${move.type}:${move.id}`;
      }
    };

    // Deterministic auto-selection: pick the candidate with the smallest
    // sort key, using Move.id as a final tie-breaker.
    const sortedCandidates = [...decisionCandidates].sort((a, b) => {
      const keyA = sortKeyForDecisionMove(a);
      const keyB = sortKeyForDecisionMove(b);

      if (keyA < keyB) return -1;
      if (keyA > keyB) return 1;

      // In extremely rare cases where the geometry-derived keys match,
      // fall back to the canonical Move.id, which is itself deterministic.
      if (a.id < b.id) return -1;
      if (a.id > b.id) return 1;
      return 0;
    });

    // Record a move rejection signal for a human decision that timed out and
    // was auto-resolved by the host rather than explicitly chosen by the
    // player. This feeds the ringrift_moves_rejected_total metric used by
    // the Rules Correctness dashboard.
    getMetricsService().recordMoveRejected('decision_timeout_auto_rejected');

    // P2 FIX: Cancel any pending WebSocket choices for this player before
    // applying the auto-resolved move. This prevents the WebSocket handler's
    // independent timeout from firing after we've already resolved the decision.
    this.wsHandler?.cancelAllChoicesForPlayer(playerSnapshot);

    const selected = sortedCandidates[0];

    let result: RulesResult;
    try {
      result = await this.rulesFacade.applyMoveById(playerSnapshot, selected.id);
    } catch (err) {
      logger.error('Failed to apply auto-resolved decision move', {
        gameId: this.gameId,
        playerNumber: playerSnapshot,
        moveId: selected.id,
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }

    if (!result.success) {
      logger.warn('Engine rejected auto-resolved decision move', {
        gameId: this.gameId,
        playerNumber: playerSnapshot,
        moveId: selected.id,
        reason: result.error,
      });
      return;
    }

    const updatedState = this.gameEngine.getGameState();
    const lastMove = updatedState.moveHistory[updatedState.moveHistory.length - 1];
    const player = updatedState.players.find((p) => p.playerNumber === playerSnapshot);

    const decisionMeta: DecisionAutoResolvedMeta = {
      choiceType: choiceTypeSnapshot,
      choiceKind: choiceKindSnapshot,
      actingPlayerNumber: playerSnapshot,
      resolvedMoveId: selected.id,
      reason: 'timeout',
    };

    if (lastMove) {
      // Attach decision auto-resolve metadata directly to the canonical Move
      // so that GamePersistenceService.serializeMoveData can persist it into
      // moveData.decisionAutoResolved for later history reconstruction.
      (lastMove as MoveWithAutoResolveMeta).decisionAutoResolved = decisionMeta;
    }

    if (lastMove && player) {
      // Persist as a normal human move attributed to the player.
      await this.persistMove(player.id, result);
    }

    const timeoutPayload: DecisionPhaseTimedOutPayload = {
      type: 'decision_phase_timed_out',
      data: {
        gameId: this.gameId,
        playerNumber: playerSnapshot,
        phase: timeoutPhase,
        autoSelectedMoveId: selected.id,
        reason: `Decision timeout: auto-selected ${selected.type}`,
      },
      timestamp: new Date().toISOString(),
    };

    this.io.to(this.gameId).emit('decision_phase_timed_out', timeoutPayload);

    await this.broadcastUpdate(result, decisionMeta);

    logger.info('Auto-resolved decision phase after timeout', {
      gameId: this.gameId,
      playerNumber: playerSnapshot,
      phase: timeoutPhase,
      moveId: selected.id,
    });

    // After an auto-resolved human move, AI may need to act next.
    await this.maybePerformAITurn();
  }

  /**
   * Start the AI watchdog timer that periodically checks for stalled AI turns.
   * This helps recover from cases where the AI turn logic silently fails.
   */
  private startAIWatchdog(): void {
    // Clear any existing watchdog
    if (this.aiWatchdogHandle) {
      clearInterval(this.aiWatchdogHandle);
    }

    // Check every 10 seconds for stalled AI turns
    this.aiWatchdogHandle = setInterval(() => {
      void this.checkAIWatchdog();
    }, 10000);

    this.lastAITurnCheck = Date.now();
  }

  /**
   * Check if an AI turn appears to be stalled and trigger recovery if needed.
   */
  private async checkAIWatchdog(): Promise<void> {
    try {
      const state = this.gameEngine.getGameState();
      if (state.gameStatus !== 'active') {
        return;
      }

      const currentPlayer = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!currentPlayer || currentPlayer.type !== 'ai') {
        // Not an AI turn - reset the check timer
        this.lastAITurnCheck = Date.now();
        return;
      }

      // Check if AI turn has been going for too long (15+ seconds without activity)
      const now = Date.now();

      // Get request age from the aiRequestState if available
      let aiRequestAge: number;
      if (
        this.aiRequestState &&
        this.aiRequestState.kind !== 'idle' &&
        'requestedAt' in this.aiRequestState
      ) {
        aiRequestAge = now - this.aiRequestState.requestedAt;
      } else {
        aiRequestAge = now - this.lastAITurnCheck;
      }

      // If no AI request is in progress and it's been >15s, something may be stuck
      const isStalled =
        (!this.aiRequestState || this.aiRequestState.kind === 'idle') && aiRequestAge > 15000;

      if (isStalled) {
        logger.warn('[AI Watchdog] AI turn appears stalled, triggering maybePerformAITurn', {
          gameId: this.gameId,
          playerNumber: state.currentPlayer,
          phase: state.currentPhase,
          aiRequestAge,
          aiRequestKind: this.aiRequestState?.kind,
        });

        this.lastAITurnCheck = now;
        await this.maybePerformAITurn();
      }
    } catch (err) {
      logger.error('[AI Watchdog] Error checking AI turn status', {
        gameId: this.gameId,
        error: (err as Error).message,
      });
    }
  }
}
