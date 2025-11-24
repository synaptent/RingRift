import { Server as SocketIOServer } from 'socket.io';
import { GameEngine } from './GameEngine';
import { RulesBackendFacade } from './RulesBackendFacade';
import { PlayerInteractionManager } from './PlayerInteractionManager';
import { WebSocketInteractionHandler } from './WebSocketInteractionHandler';
import { DelegatingInteractionHandler } from './DelegatingInteractionHandler';
import { AIInteractionHandler } from './ai/AIInteractionHandler';
import { globalAIEngine } from './ai/AIEngine';
import { getOrCreateAIUser } from '../services/AIUserService';
import { PythonRulesClient } from '../services/PythonRulesClient';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import { gameMoveLatencyHistogram } from '../utils/rulesParityMetrics';
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
import type { ClientToServerEvents, ServerToClientEvents } from '../../shared/types/websocket';
import { hashGameState } from '../../shared/engine/core';
import { SeededRNG, generateGameSeed } from '../../shared/utils/rng';

export class GameSession {
  public readonly gameId: string;
  private io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>;
  private gameEngine!: GameEngine;
  private rulesFacade!: RulesBackendFacade;
  private interactionManager!: PlayerInteractionManager;
  private wsHandler!: WebSocketInteractionHandler;
  private pythonRulesClient: PythonRulesClient;
  private userSockets: Map<string, string>; // userId -> socketId
  private rng: SeededRNG; // Per-game RNG for deterministic AI behavior
  private turnTimerId: NodeJS.Timeout | null = null; // Timer for turn countdown

  /**
   * Per-game view of AI/rules degraded-mode diagnostics. These counters are
   * derived from AIEngine and RulesBackendFacade diagnostics and are used for
   * observability and tests.
   */
  private aiQualityMode: 'normal' | 'fallbackLocalAI' | 'rulesServiceDegraded' = 'normal';
  private aiServiceFailureCount = 0;
  private aiFallbackMoveCount = 0;
  private rulesServiceFailureCount = 0;
  private rulesShadowErrorCount = 0;

  constructor(
    gameId: string,
    io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>,
    pythonRulesClient: PythonRulesClient,
    userSockets: Map<string, string>
  ) {
    this.gameId = gameId;
    this.io = io;
    this.pythonRulesClient = pythonRulesClient;
    this.userSockets = userSockets;
    // Initialize with a temporary seed; will be updated from gameState in initialize()
    this.rng = new SeededRNG(generateGameSeed());
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

    // Initialize RNG from game's seed (if available) or persist a new one
    let gameSeed: number | null = (game as any).rngSeed ?? null;

    if (typeof gameSeed !== 'number') {
      gameSeed = generateGameSeed();

      try {
        await prisma.game.update({
          where: { id: this.gameId },
          data: { rngSeed: gameSeed },
        });
      } catch (err) {
        logger.error('Failed to persist generated RNG seed for game', {
          gameId: this.gameId,
          error: (err as Error).message,
        });
      }
    }

    this.rng = new SeededRNG(gameSeed);

    logger.info('GameSession RNG initialized', {
      gameId: this.gameId,
      seed: gameSeed,
    });

    // Create game engine
    this.gameEngine = new GameEngine(
      this.gameId,
      game.boardType as keyof typeof BOARD_CONFIGS,
      players,
      timeControl,
      (game as any).isRated ?? true,
      this.interactionManager,
      gameSeed ?? undefined
    );

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
          logger.info('Auto-started game', { gameId: this.gameId, playerCount: players.length });
        } catch (err) {
          logger.error('Failed to auto-start game', {
            gameId: this.gameId,
            error: (err as Error).message,
          });
        }
      }
    }
  }

  private createPlayer(
    userData: { id: string; username: string },
    playerNumber: number,
    boardConfig: any,
    initialTimeMs: number
  ): Player {
    return {
      id: userData.id,
      username: userData.username,
      playerNumber,
      type: 'human',
      isReady: true,
      timeRemaining: initialTimeMs,
      ringsInHand: boardConfig.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    };
  }

  private replayMove(move: any) {
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

    const gameMove: Move = {
      id: move.id,
      type: move.moveType as any,
      player: parseInt(move.playerId),
      ...(from ? { from } : {}),
      to,
      timestamp: move.timestamp,
      thinkTime: 0,
      moveNumber: move.moveNumber,
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

    // Measure core move-processing latency (rules engine + game engine) for
    // successful applications. Labels are low-cardinality: boardType and phase.
    const boardTypeLabel = (currentState.boardType ?? 'unknown') as string;
    const phaseLabel = (currentState.currentPhase ?? 'unknown') as string;
    const startTime = performance.now();

    const result = await this.rulesFacade.applyMove(engineMove);
    const duration = performance.now() - startTime;
    gameMoveLatencyHistogram.labels(boardTypeLabel, phaseLabel).observe(duration);

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
    // Refresh degraded-mode diagnostics after a successful human move so
    // rules-service failures (in python or shadow modes) are visible.
    this.updateDiagnostics(player.playerNumber);
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

    const boardTypeLabel = (currentState.boardType ?? 'unknown') as string;
    const phaseLabel = (currentState.currentPhase ?? 'unknown') as string;
    const startTime = performance.now();

    const result = await this.rulesFacade.applyMoveById(player.playerNumber, moveId);
    const duration = performance.now() - startTime;
    gameMoveLatencyHistogram.labels(boardTypeLabel, phaseLabel).observe(duration);

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
    // Refresh degraded-mode diagnostics after a successful human move
    // selected by id so rules-service failures are visible.
    this.updateDiagnostics(player.playerNumber);
    await this.maybePerformAITurn();
  }

  private async persistMove(playerId: string, moveData: any, result: any) {
    const prisma = getDatabaseClient();
    if (!prisma) return;

    await prisma.move.create({
      data: {
        gameId: this.gameId,
        playerId,
        moveNumber: moveData.moveNumber,
        position: moveData.position,
        moveType: moveData.moveType,
        timestamp: new Date(),
      },
    });

    if (result.gameResult) {
      const updatedState = this.gameEngine.getGameState();
      const winnerPlayerNumber = result.gameResult.winner;
      let winnerId: string | null = null;

      if (winnerPlayerNumber !== undefined) {
        const winnerPlayer = updatedState.players.find(
          (p) => p.playerNumber === winnerPlayerNumber && p.type === 'human'
        );
        winnerId = winnerPlayer?.id ?? null;
      }

      await prisma.game.update({
        where: { id: this.gameId },
        data: {
          status: 'completed' as any,
          winnerId: winnerId ?? null,
          endedAt: new Date(),
          updatedAt: new Date(),
        },
      });
    }
  }

  private async broadcastUpdate(result: any) {
    const updatedState = this.gameEngine.getGameState();

    if (result.gameResult) {
      // Stop timer when game ends
      this.stopTurnTimer();

      this.io.to(this.gameId).emit('game_over', {
        type: 'game_over',
        data: {
          gameId: this.gameId,
          gameState: updatedState,
          gameResult: result.gameResult,
        },
        timestamp: new Date().toISOString(),
      });
    } else {
      // Start timer for the new current player
      this.startTurnTimer();

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
    }
  }

  private async maybePerformAITurn(): Promise<void> {
    try {
      const initialState = this.gameEngine.getGameState();
      if (initialState.gameStatus !== 'active') return;

      const currentPlayerNumber = initialState.currentPlayer;
      const currentPlayer = initialState.players.find(
        (p) => p.playerNumber === currentPlayerNumber
      );

      if (!currentPlayer || currentPlayer.type !== 'ai') return;

      let aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      if (!aiConfig) {
        const difficulty = currentPlayer.aiDifficulty ?? 5;
        globalAIEngine.createAI(currentPlayerNumber, difficulty);
        aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      }

      let appliedMoveType: Move['type'] | undefined;

      // Non-move decision phases (line/territory processing) always use
      // local heuristics and should only emit already-valid moves from
      // GameEngine.getValidMoves.
      if (
        initialState.currentPhase === 'line_processing' ||
        initialState.currentPhase === 'territory_processing'
      ) {
        const allCandidates = this.gameEngine.getValidMoves(currentPlayerNumber);
        const decisionCandidates = allCandidates.filter((m) => {
          if (initialState.currentPhase === 'line_processing') {
            return m.type === 'process_line' || m.type === 'choose_line_reward';
          }
          return m.type === 'process_territory_region' || m.type === 'eliminate_rings_from_stack';
        });

        if (decisionCandidates.length === 0) {
          return;
        }

        const selected = globalAIEngine.chooseLocalMoveFromCandidates(
          currentPlayerNumber,
          initialState,
          decisionCandidates,
          () => this.rng.next()
        );

        if (!selected) return;

        const { id, timestamp, moveNumber, ...rest } = selected as any;
        const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
        appliedMoveType = engineMove.type;

        const result = await this.rulesFacade.applyMove(engineMove);
        await this.handleAIMoveResult(result, currentPlayerNumber, appliedMoveType);
        return;
      }

      // Move-generating AI path (Python service first, then local fallback).
      const MAX_SERVICE_RETRIES = 2;
      let stateForAI = initialState;
      let lastError: string | undefined;

      for (let attempt = 0; attempt <= MAX_SERVICE_RETRIES; attempt++) {
        const aiMove = await globalAIEngine.getAIMove(currentPlayerNumber, stateForAI, () =>
          this.rng.next()
        );
        if (!aiMove) {
          lastError =
            lastError ?? 'AI service returned no move after retries (see earlier logs for details)';
          break;
        }

        const { id, timestamp, moveNumber, ...rest } = aiMove;
        const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
        appliedMoveType = engineMove.type;

        const result = await this.rulesFacade.applyMove(engineMove);

        if (result.success) {
          await this.handleAIMoveResult(result, currentPlayerNumber, appliedMoveType);
          return;
        }

        lastError = result.error ?? 'rules engine rejected AI service move';

        logger.error('AI service move rejected by rules engine', {
          gameId: this.gameId,
          playerNumber: currentPlayerNumber,
          moveType: engineMove.type,
          reason: result.error,
          attempt,
          stateHash: hashGameState(stateForAI),
        });

        // Re-acquire the latest game state before retrying the AI service.
        stateForAI = this.gameEngine.getGameState();
      }

      // Local fallback heuristic when the AI service repeatedly fails.
      const fallbackMove = globalAIEngine.getLocalFallbackMove(
        currentPlayerNumber,
        stateForAI,
        () => this.rng.next()
      );

      if (fallbackMove) {
        const { id, timestamp, moveNumber, ...rest } = fallbackMove as any;
        const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
        appliedMoveType = engineMove.type;

        logger.warn('Using local fallback AI move after service failure', {
          gameId: this.gameId,
          playerNumber: currentPlayerNumber,
          moveType: engineMove.type,
          stateHash: hashGameState(stateForAI),
        });

        const result = await this.rulesFacade.applyMove(engineMove);
        if (result.success) {
          await this.handleAIMoveResult(result, currentPlayerNumber, appliedMoveType);
          return;
        }

        logger.error('Local fallback AI move was rejected by rules engine', {
          gameId: this.gameId,
          playerNumber: currentPlayerNumber,
          moveType: engineMove.type,
          reason: result.error,
          stateHash: hashGameState(this.gameEngine.getGameState()),
        });

        await this.handleAIFatalFailure(currentPlayerNumber, {
          reason: result.error ?? 'Local fallback AI move rejected after AI service failures',
        });
        return;
      }

      // No valid local fallback move exists; abandon the game with a clear
      // failure reason so that operators can investigate.
      await this.handleAIFatalFailure(currentPlayerNumber, {
        reason:
          lastError ?? 'AI service produced no valid moves and local fallback had no candidates',
      });
    } catch (error) {
      logger.error('Error during AI turn', {
        gameId: this.gameId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async handleAIMoveResult(
    result: { success: boolean; error?: string; gameState?: GameState; gameResult?: any },
    currentPlayerNumber: number,
    appliedMoveType: Move['type'] | undefined
  ): Promise<void> {
    if (!result.success) {
      logger.warn('Engine rejected AI move', {
        gameId: this.gameId,
        playerNumber: currentPlayerNumber,
        reason: result.error,
      });
      return;
    }

    const updatedState = this.gameEngine.getGameState();
    const prisma = getDatabaseClient();

    if (prisma) {
      try {
        const aiUser = await getOrCreateAIUser();
        const lastMove = updatedState.moveHistory[updatedState.moveHistory.length - 1];

        if (lastMove) {
          await prisma.move.create({
            data: {
              gameId: this.gameId,
              playerId: aiUser.id,
              moveNumber: lastMove.moveNumber,
              position: JSON.stringify({ from: lastMove.from, to: lastMove.to }),
              moveType: lastMove.type as any,
              timestamp: lastMove.timestamp,
            },
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

          await prisma.game.update({
            where: { id: this.gameId },
            data: {
              status: 'completed' as any,
              winnerId: winnerId ?? null,
              endedAt: new Date(),
              updatedAt: new Date(),
            },
          });
        }
      } catch (err) {
        logger.error('Failed to persist AI move', {
          gameId: this.gameId,
          playerNumber: currentPlayerNumber,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    await this.broadcastUpdate(result);

    // Refresh per-game diagnostics now that both AI and rules backends have
    // had a chance to update their internal counters.
    this.updateDiagnostics(currentPlayerNumber);

    logger.info('AI move processed and applied', {
      gameId: this.gameId,
      playerNumber: currentPlayerNumber,
      moveType: appliedMoveType,
    });
  }

  private async handleAIFatalFailure(
    currentPlayerNumber: number,
    context: { reason: string }
  ): Promise<void> {
    const updatedState = this.gameEngine.getGameState();
    const prisma = getDatabaseClient();

    if (prisma) {
      try {
        await prisma.game.update({
          where: { id: this.gameId },
          data: {
            status: 'completed' as any,
            winnerId: null,
            endedAt: new Date(),
            updatedAt: new Date(),
          },
        });
      } catch (err) {
        logger.error('Failed to mark game as completed after AI failure', {
          gameId: this.gameId,
          playerNumber: currentPlayerNumber,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    // Construct a minimal GameResult with an abandonment reason so that
    // clients can surface a clear failure mode. We approximate final
    // scores from the current GameState; this path is expected to be
    // extremely rare and primarily diagnostic.
    const ringsEliminated: { [playerNumber: number]: number } = {};
    const territorySpaces: { [playerNumber: number]: number } = {};
    const ringsRemaining: { [playerNumber: number]: number } = {};

    for (const player of updatedState.players) {
      const num = player.playerNumber;
      ringsEliminated[num] = player.eliminatedRings;
      territorySpaces[num] = player.territorySpaces;
      ringsRemaining[num] = player.ringsInHand;
    }

    const gameResult: GameResult = {
      reason: 'abandonment',
      finalScore: {
        ringsEliminated,
        territorySpaces,
        ringsRemaining,
      },
    };

    // Emit game_error for active players to show error UI
    this.io.to(this.gameId).emit('game_error', {
      type: 'game_error',
      data: {
        message: 'AI encountered a fatal error. Game cannot continue.',
        technical: context.reason,
        gameId: this.gameId,
      },
      timestamp: new Date().toISOString(),
    });

    this.io.to(this.gameId).emit('game_over', {
      type: 'game_over',
      data: {
        gameId: this.gameId,
        gameState: updatedState,
        gameResult,
      },
      timestamp: new Date().toISOString(),
    });

    // Ensure degraded-mode diagnostics reflect the fatal failure state.
    this.updateDiagnostics(currentPlayerNumber);

    logger.error('AI turn failed after service and local fallbacks; game abandoned', {
      gameId: this.gameId,
      playerNumber: currentPlayerNumber,
      ...context,
    });
  }

  /**
   * Public accessor used by tests and observability tooling to inspect the
   * per-game degraded-mode diagnostics without exposing the internal mutable
   * fields directly.
   */
  public getAIDiagnosticsSnapshotForTesting(): {
    aiQualityMode: 'normal' | 'fallbackLocalAI' | 'rulesServiceDegraded';
    aiServiceFailureCount: number;
    aiFallbackMoveCount: number;
    rulesServiceFailureCount: number;
    rulesShadowErrorCount: number;
  } {
    return {
      aiQualityMode: this.aiQualityMode,
      aiServiceFailureCount: this.aiServiceFailureCount,
      aiFallbackMoveCount: this.aiFallbackMoveCount,
      rulesServiceFailureCount: this.rulesServiceFailureCount,
      rulesShadowErrorCount: this.rulesShadowErrorCount,
    };
  }

  /**
   * Internal helper: refresh this session's view of AI and rules degraded-mode
   * diagnostics from the underlying AIEngine and RulesBackendFacade. The
   * aiQualityMode flag is escalated based on the presence of local fallbacks
   * and rules-service failures but is never downgraded within a session.
   */
  private updateDiagnostics(currentPlayerNumber?: number): void {
    // Refresh rules diagnostics first; rules degradation is treated as
    // strictly worse than local AI fallback for quality-mode purposes.
    if (this.rulesFacade && (this.rulesFacade as any).getDiagnostics) {
      const rulesDiag = this.rulesFacade.getDiagnostics();
      const rulesFailures = rulesDiag.pythonEvalFailures + rulesDiag.pythonBackendFallbacks;

      this.rulesServiceFailureCount = rulesFailures;
      this.rulesShadowErrorCount = rulesDiag.pythonShadowErrors;

      if (rulesFailures > 0) {
        this.aiQualityMode = 'rulesServiceDegraded';
      }
    }

    if (currentPlayerNumber !== undefined) {
      const aiDiag = globalAIEngine.getDiagnostics(currentPlayerNumber);
      if (aiDiag) {
        this.aiServiceFailureCount = aiDiag.serviceFailureCount;
        this.aiFallbackMoveCount = aiDiag.localFallbackCount;

        if (this.aiQualityMode === 'normal' && this.aiFallbackMoveCount > 0) {
          this.aiQualityMode = 'fallbackLocalAI';
        }
      }
    }
  }

  /**
   * Start the turn timer for the current player
   */
  private startTurnTimer(): void {
    if (!this.gameEngine) return;

    const gameState = this.gameEngine.getGameState();
    if (!gameState.timeControl) return;

    const currentPlayer = gameState.players.find((p) => p.playerNumber === gameState.currentPlayer);
    if (!currentPlayer) return;

    // Clear any existing timer
    if (this.turnTimerId) {
      clearInterval(this.turnTimerId);
      this.turnTimerId = null;
    }

    // Update time every second and broadcast to clients
    this.turnTimerId = setInterval(() => {
      const latestState = this.gameEngine.getGameState();
      const player = latestState.players.find((p) => p.playerNumber === latestState.currentPlayer);

      if (!player) {
        this.stopTurnTimer();
        return;
      }

      // Decrement player's time
      player.timeRemaining = Math.max(0, (player.timeRemaining ?? 0) - 1000);

      // Broadcast time update to all clients
      this.io.to(this.gameId).emit('time_update', {
        playerId: player.id,
        playerNumber: player.playerNumber,
        timeRemaining: player.timeRemaining,
      });

      // Handle time expiration
      if (player.timeRemaining === 0) {
        this.handleTimeExpiration(player.playerNumber);
      }
    }, 1000);
  }

  /**
   * Stop the turn timer
   */
  private stopTurnTimer(): void {
    if (this.turnTimerId) {
      clearInterval(this.turnTimerId);
      this.turnTimerId = null;
    }
  }

  /**
   * Handle when a player runs out of time
   */
  private async handleTimeExpiration(playerNumber: number): Promise<void> {
    this.stopTurnTimer();

    logger.warn('Player time expired', {
      gameId: this.gameId,
      playerNumber,
    });

    const gameState = this.gameEngine.getGameState();

    // For AI players, just make a move immediately
    const player = gameState.players.find((p) => p.playerNumber === playerNumber);
    if (player?.type === 'ai') {
      await this.maybePerformAITurn();
      return;
    }

    // For human players, try to make a random valid move
    const validMoves = this.gameEngine.getValidMoves(playerNumber);
    if (validMoves.length > 0) {
      const randomMove = validMoves[Math.floor(this.rng.next() * validMoves.length)];
      const { id, timestamp, moveNumber, ...rest } = randomMove as any;
      const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;

      try {
        const result = await this.rulesFacade.applyMove(engineMove);
        if (result.success) {
          await this.broadcastUpdate(result);
          await this.maybePerformAITurn();
        }
      } catch (err) {
        logger.error('Failed to apply time-expiration move', {
          gameId: this.gameId,
          playerNumber,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  }
}
