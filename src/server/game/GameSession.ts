import { Server as SocketIOServer } from 'socket.io';
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
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import { getMetricsService } from '../services/MetricsService';
import { orchestratorRollout, EngineSelection } from '../services/OrchestratorRolloutService';
import { shadowComparator } from '../services/ShadowModeComparator';
import { createSimpleAdapter, createAutoSelectDecisionHandler } from './turn/TurnEngineAdapter';
import { config } from '../config';
import {
  Move,
  Player,
  GameState,
  Position,
  AIProfile,
  BOARD_CONFIGS,
  TimeControl,
  GameResult,
} from '../../shared/types/game';
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

  // Aggregated diagnostics
  private diagnosticsSnapshot: SessionAIDiagnostics = {
    rulesServiceFailureCount: 0,
    rulesShadowErrorCount: 0,
    aiServiceFailureCount: 0,
    aiFallbackMoveCount: 0,
    aiQualityMode: 'normal',
  };

  // Default AI request timeout (can be overridden via config)
  private readonly aiRequestTimeoutMs: number;

  // Engine selection for this session (legacy, orchestrator, or shadow)
  private engineSelection: EngineSelection = EngineSelection.LEGACY;

  constructor(
    gameId: string,
    io: SocketIOServer,
    pythonRulesClient: PythonRulesClient,
    userSockets: Map<string, string>
  ) {
    this.gameId = gameId;
    this.io = io;
    this.pythonRulesClient = pythonRulesClient;
    this.userSockets = userSockets;
    this.aiRequestTimeoutMs = config.aiService?.requestTimeoutMs ?? 30000;
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

    // Create players array
    const players: Player[] = [];
    const boardConfig = BOARD_CONFIGS[game.boardType as keyof typeof BOARD_CONFIGS];
    const initialTimeMs =
      typeof game.timeControl === 'string'
        ? (JSON.parse(game.timeControl).initialTime as number)
        : 600000;

    if (game.player1) {
      players.push(this.createPlayer(game.player1, 1, boardConfig, initialTimeMs));
    }
    if (game.player2) {
      players.push(this.createPlayer(game.player2, 2, boardConfig, initialTimeMs));
    }

    // Optional AI opponents
    const gameStateSnapshot = (game.gameState || {}) as any;
    const aiOpponents = gameStateSnapshot.aiOpponents as
      | {
          count: number;
          difficulty: number[];
          mode?: 'local_heuristic' | 'service';
          aiType?: 'random' | 'heuristic' | 'minimax' | 'mcts';
        }
      | undefined;

    if (aiOpponents && aiOpponents.count > 0) {
      const startingNumber = players.length + 1;
      const maxSlots = game.maxPlayers ?? 2;
      const aiCount = Math.min(aiOpponents.count, maxSlots - players.length);

      for (let i = 0; i < aiCount; i++) {
        const playerNumber = startingNumber + i;
        const difficulty = aiOpponents.difficulty?.[i] ?? 5;
        const aiProfile: AIProfile = {
          difficulty,
          mode: aiOpponents.mode ?? 'service',
          ...(aiOpponents.aiType && { aiType: aiOpponents.aiType }),
        };

        players.push({
          id: `ai-${this.gameId}-${playerNumber}`,
          username: `AI (Level ${difficulty})`,
          playerNumber,
          type: 'ai',
          isReady: true,
          timeRemaining: initialTimeMs,
          ringsInHand: boardConfig.ringsPerPlayer,
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
    const aiHandler = new AIInteractionHandler();
    const delegatingHandler = new DelegatingInteractionHandler(
      this.wsHandler,
      aiHandler,
      (playerNumber: number) => {
        const player = players.find((p) => p.playerNumber === playerNumber);
        return player?.type ?? 'human';
      }
    );

    this.interactionManager = new PlayerInteractionManager(delegatingHandler);

    // Create game engine
    this.gameEngine = new GameEngine(
      this.gameId,
      game.boardType as keyof typeof BOARD_CONFIGS,
      players,
      timeControl,
      (game as any).isRated ?? true,
      this.interactionManager
    );

    // Decide which engine path to use for this session based on
    // orchestrator rollout feature flags and user targeting.
    this.configureEngineSelection(players);

    this.gameEngine.enableMoveDrivenDecisionPhases();

    // Replay moves
    for (const move of game.moves) {
      this.replayMove(move);
    }

    this.rulesFacade = new RulesBackendFacade(this.gameEngine, this.pythonRulesClient);

    // Auto-start logic
    if ((game.status as any) === 'waiting' && players.length >= (game.maxPlayers ?? 2)) {
      const allReady = players.every((p) => p.isReady);
      if (allReady) {
        try {
          await prisma.game.update({
            where: { id: this.gameId },
            data: {
              status: 'active' as any,
              startedAt: new Date(),
            },
          });
          this.gameEngine.startGame();
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

    logger.info('GameSession initialized', {
      gameId: this.gameId,
      status: this.sessionStatus?.kind,
      playerCount: players.length,
    });
  }

  private createPlayer(
    dbPlayer: { id: string; username: string | null },
    playerNumber: number,
    boardConfig: { ringsPerPlayer: number },
    initialTimeMs: number
  ): Player {
    return {
      id: dbPlayer.id,
      username: dbPlayer.username ?? `Player ${playerNumber}`,
      playerNumber,
      type: 'human',
      isReady: true,
      timeRemaining: initialTimeMs,
      ringsInHand: boardConfig.ringsPerPlayer,
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
   * See docs/ORCHESTRATOR_ROLLOUT_PLAN.md (Phase A – Backend orchestrator-only).
   */
  private configureEngineSelection(players: Player[]): void {
    // Prefer a human player for targeting; fall back to undefined.
    const primaryHuman = players.find((p) => p.type === 'human');
    const userId = primaryHuman?.id;

    const decision = orchestratorRollout.selectEngine(this.gameId, userId);
    this.engineSelection = decision.engine;

    // Adapter enablement is now controlled globally via
    // config.featureFlags.orchestrator.adapterEnabled and is no longer
    // overridden per session. EngineSelection is retained for observability
    // and for selecting shadow-comparison behaviour in higher-level services.

    // Record a rollout session selection metric for observability.
    getMetricsService().recordOrchestratorSession(decision.engine, decision.reason);

    logger.info('Engine selection for game session', {
      gameId: this.gameId,
      engine: decision.engine,
      reason: decision.reason,
      targetedUserId: userId,
    });
  }

  /**
   * Apply a move when this session is running in SHADOW mode: the legacy
   * GameEngine/RulesBackendFacade path remains authoritative, while the
   * orchestrator adapter runs in parallel on a cloned state via
   * ShadowModeComparator for comparison and metrics.
   */
  private async applyMoveWithOrchestratorShadow(
    preState: GameState,
    engineMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
  ): Promise<RulesResult> {
    const moveNumber = preState.moveHistory.length + 1;

    const { result } = await shadowComparator.compare(
      this.gameId,
      moveNumber,
      async () => this.rulesFacade.applyMove(engineMove),
      async () => this.runOrchestratorShadow(preState, engineMove, moveNumber)
    );

    return result as RulesResult;
  }

  /**
   * Run the shared orchestrator adapter on a cloned snapshot of the
   * pre-move GameState for shadow comparison. This never mutates the
   * authoritative GameEngine instance.
   */
  private async runOrchestratorShadow(
    preState: GameState,
    engineMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>,
    moveNumber: number
  ): Promise<RulesResult> {
    const fullMove: Move = {
      ...engineMove,
      id: `shadow-${moveNumber}`,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    };

    // Use the simple in-memory adapter so the orchestrator operates on a
    // detached copy of the GameState for shadow evaluation.
    const decisionHandler = createAutoSelectDecisionHandler();
    const { adapter, getState } = createSimpleAdapter(preState, decisionHandler);
    const adapterResult = await adapter.processMove(fullMove);
    const nextState = getState();

    const result: RulesResult = {
      success: adapterResult.success,
      gameState: nextState,
    };

    if (adapterResult.error) {
      result.error = adapterResult.error;
    }
    if (adapterResult.victoryResult) {
      result.gameResult = adapterResult.victoryResult;
    }

    return result;
  }

  private replayMove(move: any): void {
    let from: Position | undefined;
    let to: Position | undefined;

    const rawPosition = move.position as unknown;
    if (typeof rawPosition === 'string') {
      try {
        const parsed = JSON.parse(rawPosition) as any;
        from = parsed.from as Position | undefined;
        to = (parsed.to as Position | undefined) ?? (parsed as Position);
      } catch (err) {
        logger.warn('Failed to parse persisted move.position string', {
          gameId: this.gameId,
          moveId: move.id,
          rawPosition,
          error: (err as Error).message,
        });
      }
    } else if (rawPosition && typeof rawPosition === 'object') {
      const parsed = rawPosition as any;
      from = parsed.from as Position | undefined;
      to = (parsed.to as Position | undefined) ?? (parsed as Position);
    }

    if (!to) {
      logger.warn('Skipping historical move with no destination', {
        gameId: this.gameId,
        moveId: move.id,
        rawPosition,
      });
      return;
    }

    const gameMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: move.moveType as any,
      player: parseInt(move.playerId),
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
   */
  updateDiagnostics(playerNumber: number): void {
    // Get rules facade diagnostics
    const rulesDiag = this.rulesFacade?.getDiagnostics?.() ?? {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    // Get AI engine diagnostics for this player
    const aiDiag: AIDiagnostics = globalAIEngine.getDiagnostics(playerNumber) ?? {
      serviceFailureCount: 0,
      localFallbackCount: 0,
    };

    this.diagnosticsSnapshot = {
      rulesServiceFailureCount: rulesDiag.pythonShadowErrors,
      rulesShadowErrorCount: rulesDiag.pythonShadowErrors,
      aiServiceFailureCount: aiDiag.serviceFailureCount,
      aiFallbackMoveCount: aiDiag.localFallbackCount,
      aiQualityMode: this.computeAIQualityMode(rulesDiag, aiDiag),
    };
  }

  private computeAIQualityMode(
    rulesDiag: { pythonShadowErrors: number },
    aiDiag: AIDiagnostics
  ): 'normal' | 'fallbackLocalAI' | 'rulesServiceDegraded' {
    if (rulesDiag.pythonShadowErrors > 0) {
      return 'rulesServiceDegraded';
    }
    if (aiDiag.localFallbackCount > 0) {
      return 'fallbackLocalAI';
    }
    return 'normal';
  }

  public async handlePlayerMove(socket: any, moveData: any): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) throw new Error('Database not available');

    const game = await prisma.game.findUnique({ where: { id: this.gameId } });
    if (!game) throw new Error('Game not found');
    if ((game.status as any) !== 'active') throw new Error('Game is not active');

    const currentState = this.gameEngine.getGameState();
    const player = currentState.players.find((p) => p.id === socket.userId);
    if (!player) {
      // Check if user is a spectator
      if (currentState.spectators.includes(socket.userId)) {
        throw new Error('Spectators cannot make moves');
      }
      throw new Error('Current socket user is not a player in this game');
    }

    let from: Position | undefined;
    let to: Position | undefined;

    if (typeof moveData.position === 'string') {
      try {
        const parsed = JSON.parse(moveData.position);
        from = parsed.from as Position | undefined;
        to = (parsed.to as Position | undefined) ?? (parsed as Position);
      } catch (err) {
        logger.warn('Failed to parse move.position payload', {
          gameId: this.gameId,
          rawPosition: moveData.position,
          error: (err as Error).message,
        });
        throw new Error('Invalid move position payload');
      }
    }

    if (!to) throw new Error('Move destination is required');

    const engineMove = {
      player: player.playerNumber,
      type: moveData.moveType as Move['type'],
      from,
      to,
      thinkTime: 0,
    } as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;

    // Capture a pre-move snapshot for shadow-mode orchestrator comparison.
    const preState = this.gameEngine.getGameState();

    const result =
      this.engineSelection === EngineSelection.SHADOW
        ? await this.applyMoveWithOrchestratorShadow(preState, engineMove)
        : await this.rulesFacade.applyMove(engineMove);
    if (!result.success) {
      logger.warn('Engine rejected move', {
        gameId: this.gameId,
        userId: socket.userId,
        reason: result.error,
      });
      throw new Error(result.error || 'Invalid move');
    }

    await this.persistMove(socket.userId, moveData, result);
    await this.broadcastUpdate(result);
    await this.maybePerformAITurn();
  }

  public async handlePlayerMoveById(socket: any, moveId: string): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) throw new Error('Database not available');

    const game = await prisma.game.findUnique({ where: { id: this.gameId } });
    if (!game) throw new Error('Game not found');
    if ((game.status as any) !== 'active') throw new Error('Game is not active');

    const currentState = this.gameEngine.getGameState();
    const player = currentState.players.find((p) => p.id === socket.userId);
    if (!player) {
      // Check if user is a spectator
      if (currentState.spectators.includes(socket.userId)) {
        throw new Error('Spectators cannot make moves');
      }
      throw new Error('Current socket user is not a player in this game');
    }

    const result = await this.rulesFacade.applyMoveById(player.playerNumber, moveId);
    if (!result.success) {
      logger.warn('Engine rejected move by id', {
        gameId: this.gameId,
        userId: socket.userId,
        moveId,
        reason: result.error,
      });
      throw new Error(result.error || 'Invalid move selection');
    }

    const updatedState = this.gameEngine.getGameState();
    const lastMove = updatedState.moveHistory[updatedState.moveHistory.length - 1];

    if (lastMove) {
      await this.persistMove(
        socket.userId,
        {
          moveNumber: lastMove.moveNumber,
          position: JSON.stringify({ from: lastMove.from, to: lastMove.to }),
          moveType: lastMove.type,
        },
        result
      );
    }

    await this.broadcastUpdate(result);
    await this.maybePerformAITurn();
  }

  private async persistMove(playerId: string, moveData: any, result: any): Promise<void> {
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
      const winnerPlayerNumber = result.gameResult.winner;
      let winnerId: string | null = null;

      if (winnerPlayerNumber !== undefined) {
        const winnerPlayer = updatedState.players.find(
          (p) => p.playerNumber === winnerPlayerNumber && p.type === 'human'
        );
        winnerId = winnerPlayer?.id ?? null;
      }

      // Use GamePersistenceService to finish the game with final state
      await GamePersistenceService.finishGame(
        this.gameId,
        winnerId,
        updatedState,
        result.gameResult
      );
    }
  }

  private async broadcastUpdate(result: any): Promise<void> {
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
    } else {
      // Broadcast state to all connected clients in the room
      // For active players, we include their valid moves
      // For spectators, validMoves is empty
      const room = this.io.sockets.adapter.rooms.get(this.gameId);
      if (room) {
        for (const socketId of room) {
          const socket = this.io.sockets.sockets.get(socketId) as any;
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

          socket.emit('game_state', {
            type: 'game_update',
            data: {
              gameId: this.gameId,
              gameState: updatedState,
              validMoves,
            },
            timestamp: new Date().toISOString(),
          });
        }
      }

      // Update session status
      this.recomputeSessionStatus(updatedState);
    }
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

      let result: { success: boolean; error?: string; gameState?: GameState; gameResult?: any };
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
              return m.type === 'process_line' || m.type === 'choose_line_reward';
            }
            return m.type === 'process_territory_region' || m.type === 'eliminate_rings_from_stack';
          });

          if (decisionCandidates.length === 0) {
            this.aiAbortController = null;
            return;
          }

          const selected = globalAIEngine.chooseLocalMoveFromCandidates(
            currentPlayerNumber,
            state,
            decisionCandidates
          );

          if (!selected) {
            this.aiAbortController = null;
            return;
          }

          const { id, timestamp, moveNumber, ...rest } = selected as any;
          const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
          appliedMoveType = engineMove.type;

          result =
            this.engineSelection === EngineSelection.SHADOW
              ? await this.applyMoveWithOrchestratorShadow(state, engineMove)
              : await this.rulesFacade.applyMove(engineMove);
        } else {
          // Get AI move with timeout
          const aiMove = await this.getAIMoveWithTimeout(
            currentPlayerNumber,
            state,
            this.aiRequestTimeoutMs
          );

          // Check if request was canceled during execution
          if (this.aiAbortController?.signal.aborted) {
            return;
          }

          if (!aiMove) {
            await this.handleNoMoveFromService(currentPlayerNumber, state);
            return;
          }

          const { id, timestamp, moveNumber, ...rest } = aiMove;
          const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
          appliedMoveType = engineMove.type;

          result =
            this.engineSelection === EngineSelection.SHADOW
              ? await this.applyMoveWithOrchestratorShadow(state, engineMove)
              : await this.rulesFacade.applyMove(engineMove);
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
        });

        // Recursively check for next AI turn
        await this.maybePerformAITurn();
      } catch (error) {
        // Check if this was a timeout
        if (isDeadlineExceeded(this.aiRequestState)) {
          this.aiRequestState = markTimedOut(this.aiRequestState);
          getMetricsService().recordAITurnRequestTerminal('timed_out');
          getMetricsService().recordAIFallback('timeout');

          logger.warn('AI request timed out', {
            gameId: this.gameId,
            playerNumber: currentPlayerNumber,
            durationMs: (this.aiRequestState as any).durationMs,
          });

          // Try local fallback after timeout
          await this.handleNoMoveFromService(currentPlayerNumber, state);
        } else {
          // Some other error
          const errorType = (error as any)?.aiErrorType ?? 'unknown';
          this.aiRequestState = markFailed('AI_SERVICE_ERROR', errorType, this.aiRequestState);

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
    timeoutMs: number
  ): Promise<Move | null> {
    const result = await runWithTimeout(() => globalAIEngine.getAIMove(playerNumber, state), {
      timeoutMs,
    });

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

    // Try local fallback
    const fallbackMove = globalAIEngine.getLocalFallbackMove(playerNumber, state);

    if (fallbackMove) {
      const result =
        this.engineSelection === EngineSelection.SHADOW
          ? await this.applyMoveWithOrchestratorShadow(state, fallbackMove)
          : await this.rulesFacade.applyMove(fallbackMove);

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

    // Try local fallback
    const fallbackMove = globalAIEngine.getLocalFallbackMove(playerNumber, state);

    if (fallbackMove) {
      const result =
        this.engineSelection === EngineSelection.SHADOW
          ? await this.applyMoveWithOrchestratorShadow(state, fallbackMove)
          : await this.rulesFacade.applyMove(fallbackMove);

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
   */
  private async handleAIFatalFailure(playerNumber: number, reason: string): Promise<void> {
    this.aiRequestState = markFailed(
      'AI_SERVICE_OVERLOADED',
      'both_service_and_fallback_failed',
      this.aiRequestState
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
  }

  /**
   * Persist AI move to database using GamePersistenceService
   */
  private async persistAIMove(
    playerNumber: number,
    moveType: Move['type'] | undefined,
    result: any
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
        const winnerPlayerNumber = result.gameResult.winner;
        let winnerId: string | null = null;

        if (winnerPlayerNumber !== undefined) {
          const winnerPlayer = updatedState.players.find(
            (p) => p.playerNumber === winnerPlayerNumber && p.type === 'human'
          );
          winnerId = winnerPlayer?.id ?? null;
        }

        // Use GamePersistenceService to finish the game with final state
        await GamePersistenceService.finishGame(
          this.gameId,
          winnerId,
          updatedState,
          result.gameResult
        );
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
    // Cancel any in-flight AI request
    this.cancelInFlightAIRequest(reason);

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
  // Decision Phase Timeout API (stub implementations for compatibility)
  // ===================

  /**
   * Get remaining time for current decision phase timeout.
   * Returns null if no timeout is active.
   */
  getDecisionPhaseRemainingMs(): number | null {
    // TODO: Implement when decision phase timeout feature is added
    return null;
  }

  /**
   * Reset/clear the current decision phase timeout.
   */
  resetDecisionPhaseTimeout(): void {
    // TODO: Implement when decision phase timeout feature is added
  }
}
