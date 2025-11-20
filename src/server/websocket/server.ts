import { Server as SocketIOServer } from 'socket.io';
import { Server as HTTPServer } from 'http';
import { Socket } from 'socket.io';
import jwt from 'jsonwebtoken';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import { GameEngine } from '../game/GameEngine';
import {
  Move,
  Player,
  TimeControl,
  BOARD_CONFIGS,
  PlayerChoiceResponse,
  Position,
  AIProfile,
} from '../../shared/types/game';
import { WebSocketInteractionHandler } from '../game/WebSocketInteractionHandler';
import { PlayerInteractionManager } from '../game/PlayerInteractionManager';
import { DelegatingInteractionHandler } from '../game/DelegatingInteractionHandler';
import { AIInteractionHandler } from '../game/ai/AIInteractionHandler';
import { globalAIEngine } from '../game/ai/AIEngine';
import { getOrCreateAIUser } from '../services/AIUserService';

export interface AuthenticatedSocket extends Socket {
  userId?: string;
  username?: string;
  gameId?: string;
}

export class WebSocketServer {
  private io: SocketIOServer;
  private gameRooms: Map<string, Set<string>> = new Map();
  private userSockets: Map<string, string> = new Map();
  private gameEngines: Map<string, GameEngine> = new Map();
  private interactionManagers: Map<string, PlayerInteractionManager> = new Map();
  private interactionHandlers: Map<string, WebSocketInteractionHandler> = new Map();

  constructor(httpServer: HTTPServer) {
    // Align WebSocket CORS with the HTTP API CORS configuration. Prefer an
    // explicit CLIENT_URL, then fall back to CORS_ORIGIN (used by Express),
    // then the first entry in ALLOWED_ORIGINS, and finally the Vite dev
    // origin http://localhost:5173. This avoids local dev failures where the
    // Socket.IO layer only allowed http://localhost:3000.
    const allowedOrigin =
      process.env.CLIENT_URL ||
      process.env.CORS_ORIGIN ||
      (process.env.ALLOWED_ORIGINS
        ? process.env.ALLOWED_ORIGINS.split(',')[0]
        : 'http://localhost:5173');

    this.io = new SocketIOServer(httpServer, {
      cors: {
        origin: allowedOrigin,
        methods: ['GET', 'POST'],
        credentials: true,
      },
      transports: ['websocket', 'polling'],
    });

    this.setupMiddleware();
    this.setupEventHandlers();
  }

  private setupMiddleware() {
    // Authentication middleware
    this.io.use(async (socket: AuthenticatedSocket, next) => {
      try {
        const token = socket.handshake.auth.token || socket.handshake.query.token;

        if (!token) {
          return next(new Error('Authentication token required'));
        }

        const secret = process.env.JWT_SECRET;
        if (!secret) {
          return next(new Error('JWT_SECRET not configured'));
        }

        const decoded = jwt.verify(token, secret) as any;

        if (!decoded.userId || !decoded.email) {
          return next(new Error('Invalid token payload'));
        }

        // Verify user exists and is active
        const prisma = getDatabaseClient();
        if (!prisma) {
          return next(new Error('Database not available'));
        }

        const user = await prisma.user.findUnique({
          where: { id: decoded.userId },
          select: {
            id: true,
            username: true,
            isActive: true,
          },
        });

        if (!user || !user.isActive) {
          return next(new Error('User not found or inactive'));
        }

        socket.userId = user.id;
        socket.username = user.username;

        logger.info('WebSocket authenticated', {
          userId: user.id,
          username: user.username,
          socketId: socket.id,
        });

        next();
      } catch (error) {
        logger.error('WebSocket authentication failed:', error);
        next(new Error('Authentication failed'));
      }
    });
  }

  private setupEventHandlers() {
    this.io.on('connection', (socket: AuthenticatedSocket) => {
      logger.info('WebSocket connected', {
        userId: socket.userId,
        username: socket.username,
        socketId: socket.id,
      });

      // Store user socket mapping
      if (socket.userId) {
        this.userSockets.set(socket.userId, socket.id);
      }

      // Join game room
      socket.on('join_game', async (data: { gameId: string }) => {
        try {
          await this.handleJoinGame(socket, data.gameId);
        } catch (error) {
          logger.error('Error joining game:', error);
          socket.emit('error', { message: 'Failed to join game' });
        }
      });

      // Leave game room
      socket.on('leave_game', async (data: { gameId: string }) => {
        try {
          await this.handleLeaveGame(socket, data.gameId);
        } catch (error) {
          logger.error('Error leaving game:', error);
          socket.emit('error', { message: 'Failed to leave game' });
        }
      });

      // Handle player moves
      socket.on('player_move', async (data: any) => {
        try {
          await this.handlePlayerMove(socket, data);
        } catch (error) {
          logger.error('Error handling player move:', error);
          socket.emit('error', { message: 'Invalid move' });
        }
      });

      // Handle chat messages
      socket.on('chat_message', async (data: any) => {
        try {
          await this.handleChatMessage(socket, data);
        } catch (error) {
          logger.error('Error handling chat message:', error);
          socket.emit('error', { message: 'Failed to send message' });
        }
      });

      // Handle player choice responses
      socket.on('player_choice_response', (response: PlayerChoiceResponse<any>) => {
        try {
          const gameId = socket.gameId;
          if (!gameId) {
            throw new Error('player_choice_response received without an active gameId');
          }

          const handler = this.interactionHandlers.get(gameId);
          if (!handler) {
            throw new Error(`No WebSocketInteractionHandler found for gameId=${gameId}`);
          }

          handler.handleChoiceResponse(response);
        } catch (error) {
          logger.error('Error handling player_choice_response', error);
          socket.emit('error', { message: 'Invalid choice response' });
        }
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        this.handleDisconnect(socket);
      });
    });
  }

  private async getOrCreateGameEngine(gameId: string): Promise<GameEngine> {
    if (this.gameEngines.has(gameId)) {
      return this.gameEngines.get(gameId)!;
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    // Load game from database
    const game = await prisma.game.findUnique({
      where: { id: gameId },
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

    // Create players array (humans + optional AI opponents from persisted gameState)
    const players: Player[] = [];
    const boardConfig = BOARD_CONFIGS[game.boardType as keyof typeof BOARD_CONFIGS];
    // timeControl is stored as a Prisma JsonValue; in practice it is
    // persisted as JSON. Only parse when it is a string; otherwise
    // fall back to a reasonable default.
    const initialTimeMs =
      typeof game.timeControl === 'string'
        ? (JSON.parse(game.timeControl).initialTime as number)
        : 600000;

    if (game.player1) {
      players.push({
        id: game.player1.id,
        username: game.player1.username,
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: initialTimeMs,
        ringsInHand: boardConfig.ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      });
    }
    if (game.player2) {
      players.push({
        id: game.player2.id,
        username: game.player2.username,
        playerNumber: 2,
        type: 'human',
        isReady: true,
        timeRemaining: initialTimeMs,
        ringsInHand: boardConfig.ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      });
    }

    // Optional AI opponents (persisted in gameState.aiOpponents). The
    // shape here mirrors CreateGameSchema.aiOpponents so that lobby UI
    // can configure difficulty, control mode, and tactical type.
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
          // Persisted mode/aiType come from CreateGameSchema.aiOpponents
          // via game.gameState.aiOpponents; default to service-backed
          // AI if not specified so behaviour remains backwards-compatible.
          mode: aiOpponents.mode ?? 'service',
          ...(aiOpponents.aiType && { aiType: aiOpponents.aiType }),
        };

        players.push({
          id: `ai-${gameId}-${playerNumber}`,
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
            gameId,
            playerNumber,
            difficulty,
            error: (err as Error).message,
          });
        }
      }
    }

    // Create time control from the persisted JSON value. Support both
    // stringified and structured JSON representations.
    let timeControl: TimeControl;
    if (typeof game.timeControl === 'string') {
      timeControl = JSON.parse(game.timeControl) as TimeControl;
    } else if (game.timeControl && typeof game.timeControl === 'object') {
      timeControl = game.timeControl as unknown as TimeControl;
    } else {
      // Default to a reasonable rapid-style time control if nothing is
      // persisted. This keeps us within the TimeControl.type union.
      timeControl = { type: 'rapid', initialTime: 600000, increment: 0 };
    }

    // Map playerNumber -> Socket.IO target (typically a user socket id).
    const getTargetForPlayer = (playerNumber: number): string | undefined => {
      const player = players.find((p) => p.playerNumber === playerNumber);
      if (!player) return undefined;
      return this.userSockets.get(player.id);
    };

    const wsHandler = new WebSocketInteractionHandler(this.io, gameId, getTargetForPlayer, 30_000);

    const aiHandler = new AIInteractionHandler();
    const delegatingHandler = new DelegatingInteractionHandler(
      wsHandler,
      aiHandler,
      (playerNumber: number) => {
        const player = players.find((p) => p.playerNumber === playerNumber);
        return player?.type ?? 'human';
      }
    );

    const interactionManager = new PlayerInteractionManager(delegatingHandler);

    // Create game engine wired to the interaction manager for choices
    const gameEngine = new GameEngine(
      gameId,
      game.boardType as keyof typeof BOARD_CONFIGS,
      players,
      timeControl,
      (game as any).isRated ?? true,
      interactionManager
    );

    // Replay moves if any exist. move.position is stored as a
    // Prisma JsonValue; historically this has been a JSON string, but
    // we also support structured JSON. Handle both shapes without
    // assuming a specific subtype.
    for (const move of game.moves) {
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
            gameId,
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

      // Historical records should always contain a destination; if we
      // cannot recover one, skip this move rather than constructing an
      // invalid Move object.
      if (!to) {
        logger.warn('Skipping historical move with no destination', {
          gameId,
          moveId: move.id,
          rawPosition,
        });
        continue;
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
        gameEngine.makeMove(gameMove);
      } catch (err) {
        logger.error('Failed to replay historical move', {
          gameId,
          moveId: move.id,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    this.gameEngines.set(gameId, gameEngine);
    this.interactionManagers.set(gameId, interactionManager);
    this.interactionHandlers.set(gameId, wsHandler);

     // Auto-start logic: If all players are present (human or AI) and the game
     // is still in 'waiting' status, mark it as ACTIVE in the database.
     // Note: GameEngine initializes with 'waiting' status but doesn't strictly
     // enforce a start transition, so we handle the DB sync here.
     if ((game.status as any) === 'waiting' && players.length >= (game.maxPlayers ?? 2)) {
       // Check if all players are ready (AI are always ready, humans might need explicit ready)
       // For now, we assume presence in the players array implies readiness for this check.
       const allReady = players.every((p) => p.isReady);
    
       if (allReady) {
         try {
           await prisma.game.update({
             where: { id: gameId },
             data: {
               status: 'active' as any,
               startedAt: new Date(),
             },
           });
    
           // Update the engine's internal state to match
           // (GameEngine.gameState is public-ish via getGameState, but we can't set it directly.
           // However, GameEngine doesn't block moves based on 'waiting' status, so this is mostly for DB sync).
           logger.info('Auto-started game', { gameId, playerCount: players.length });
         } catch (err) {
           logger.error('Failed to auto-start game', { gameId, error: (err as Error).message });
         }
       }
     }

    return gameEngine;
  }

  private async handleJoinGame(socket: AuthenticatedSocket, gameId: string) {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    // Verify game exists and user has access
    const game = await prisma.game.findUnique({
      where: { id: gameId },
      include: {
        player1: { select: { id: true, username: true } },
        player2: { select: { id: true, username: true } },
        player3: { select: { id: true, username: true } },
        player4: { select: { id: true, username: true } },
      },
    });

    if (!game) {
      throw new Error('Game not found');
    }

    const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
      Boolean
    );

    const isPlayer = playerIds.includes(socket.userId!);
    const canSpectate = game.allowSpectators;

    if (!isPlayer && !canSpectate) {
      throw new Error('Access denied');
    }

    // Get or create game engine
    const gameEngine = await this.getOrCreateGameEngine(gameId);

    // Join the game room
    socket.join(gameId);
    socket.gameId = gameId;

    // Add to game room tracking
    if (!this.gameRooms.has(gameId)) {
      this.gameRooms.set(gameId, new Set());
    }
    this.gameRooms.get(gameId)!.add(socket.id);

    // Send current game state with full RingRift state
    socket.emit('game_state', {
      type: 'game_update',
      data: {
        gameId,
        gameState: gameEngine.getGameState(),
        validMoves: isPlayer
          ? gameEngine.getValidMoves(gameEngine.getGameState().currentPlayer)
          : [],
      },
      timestamp: new Date().toISOString(),
    });

    // Notify others in the room
    socket.to(gameId).emit('player_joined', {
      type: 'player_joined',
      data: {
        gameId,
        player: {
          id: socket.userId!,
          username: socket.username!,
        },
      },
      timestamp: new Date().toISOString(),
    });

    logger.info('Player joined game room', {
      userId: socket.userId,
      gameId,
      isPlayer,
      canSpectate,
      gamePhase: gameEngine.getGameState().currentPhase,
      currentPlayer: gameEngine.getGameState().currentPlayer,
    });
  }

  private async handleLeaveGame(socket: AuthenticatedSocket, gameId: string) {
    socket.leave(gameId);

    // Remove from game room tracking
    if (this.gameRooms.has(gameId)) {
      this.gameRooms.get(gameId)!.delete(socket.id);
      if (this.gameRooms.get(gameId)!.size === 0) {
        this.gameRooms.delete(gameId);
      }
    }

    // Notify others in the room
    socket.to(gameId).emit('player_left', {
      type: 'player_left',
      data: {
        gameId,
        player: {
          id: socket.userId!,
          username: socket.username!,
        },
      },
      timestamp: new Date().toISOString(),
    });

    delete socket.gameId;

    logger.info('Player left game room', {
      userId: socket.userId,
      gameId,
    });
  }

  private async handlePlayerMove(socket: AuthenticatedSocket, data: any) {
    const { gameId, move } = data;

    if (!socket.gameId || socket.gameId !== gameId) {
      throw new Error('Not in game room');
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    // Verify game exists and is active
    const game = await prisma.game.findUnique({
      where: { id: gameId },
    });

    if (!game) {
      throw new Error('Game not found');
    }

    if ((game.status as any) !== 'active') {
      throw new Error('Game is not active');
    }

    // Resolve GameEngine instance for this game (cached after first load)
    const gameEngine = await this.getOrCreateGameEngine(gameId);
    const currentState = gameEngine.getGameState();

    // Determine the numeric playerNumber for this socket's user based on
    // the engine's authoritative player list.
    const player = currentState.players.find((p) => p.id === socket.userId);
    if (!player) {
      throw new Error('Current socket user is not a player in this game');
    }

    // Parse the client-supplied position payload. The current client
    // format is a JSON stringified object of the form { from, to }.
    let from: Position | undefined;
    let to: Position | undefined;

    if (typeof move.position === 'string') {
      try {
        const parsed = JSON.parse(move.position);
        from = parsed.from as Position | undefined;
        to = (parsed.to as Position | undefined) ?? (parsed as Position);
      } catch (err) {
        logger.warn('Failed to parse move.position payload', {
          gameId,
          rawPosition: move.position,
          error: (err as Error).message,
        });
        throw new Error('Invalid move position payload');
      }
    }

    // At minimum we require a destination position; the client is expected
    // to provide this in the current payload shape.
    if (!to) {
      throw new Error('Move destination is required');
    }

    // Construct a partial Move for GameEngine. For now we support simple
    // non-capture movement; capture-specific fields (captureTarget,
    // buildAmount, etc.) can be added as the client grows more capable.
    const engineMove = {
      player: player.playerNumber,
      type: move.moveType as Move['type'],
      from,
      to,
      thinkTime: 0,
    } as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;

    // Ask the engine to apply the move. If invalid, surface an error back
    // to the client instead of blindly broadcasting.
    const result = await gameEngine.makeMove(engineMove);
    if (!result.success) {
      logger.warn('Engine rejected move', {
        gameId,
        userId: socket.userId,
        reason: result.error,
      });
      throw new Error(result.error || 'Invalid move');
    }

    const updatedState = gameEngine.getGameState();

    // Persist the move as-is for now, keeping the existing representation
    // (moveNumber and position JSON). In the future we may migrate to
    // storing the richer Move shape directly.
    await prisma.move.create({
      data: {
        gameId,
        playerId: socket.userId!,
        moveNumber: move.moveNumber,
        position: move.position,
        moveType: move.moveType,
        timestamp: new Date(),
      },
    });

    // If this move ended the game, persist the terminal status and emit
    // a dedicated game_over event instead of a normal game_state
    // update. The client can use this to present a clear victory UI.
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
        where: { id: gameId },
        data: {
          status: 'completed' as any,
          winnerId: winnerId ?? null,
          endedAt: new Date(),
          updatedAt: new Date(),
        },
      });

      this.io.to(gameId).emit('game_over', {
        type: 'game_over',
        data: {
          gameId,
          gameState: updatedState,
          gameResult: result.gameResult,
        },
        timestamp: new Date().toISOString(),
      });

      logger.info('Game ended after human move', {
        gameId,
        winnerPlayerNumber,
        reason: result.gameResult.reason,
      });

      return;
    }

    // Broadcast the updated game state to all participants. The client
    // already hydrates BoardState and uses this as the single source of
    // truth when rendering.
    this.io.to(gameId).emit('game_state', {
      type: 'game_update',
      data: {
        gameId,
        gameState: updatedState,
        validMoves: gameEngine.getValidMoves(updatedState.currentPlayer),
      },
      timestamp: new Date().toISOString(),
    });

    logger.info('Player move processed and applied', {
      userId: socket.userId,
      gameId,
      moveType: move.moveType,
      playerNumber: player.playerNumber,
    });

    // After a human move, if the next player is AI, let the AI service
    // select and apply a move via the AIEngine. This keeps the
    // GameEngine as the source of truth and uses the same broadcast
    // pipeline as human moves.
    await this.maybePerformAITurn(gameId, gameEngine);
  }

  /**
   * If the current player in the given game is AI-controlled, request a
   * move from the Python AI service via globalAIEngine and apply it via
   * GameEngine. The resulting state is broadcast to all participants.
   *
   * This is intentionally conservative: it performs at most one AI move
   * per call and bails out on any error, logging instead of throwing, to
   * avoid destabilising the WebSocket loop.
   */
  private async maybePerformAITurn(gameId: string, gameEngine: GameEngine): Promise<void> {
    try {
      const state = gameEngine.getGameState();

      if (state.gameStatus !== 'active') {
        return;
      }

      const currentPlayerNumber = state.currentPlayer;
      const currentPlayer = state.players.find((p) => p.playerNumber === currentPlayerNumber);

      if (!currentPlayer || currentPlayer.type !== 'ai') {
        return;
      }

      const aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      if (!aiConfig) {
        // If no AI config exists, attempt to lazily create one using
        // the player's aiDifficulty or a reasonable default.
        const difficulty = currentPlayer.aiDifficulty ?? 5;
        globalAIEngine.createAI(currentPlayerNumber, difficulty);
      }

      const aiMove = await globalAIEngine.getAIMove(currentPlayerNumber, state);
      if (!aiMove) {
        logger.warn('AI did not return a move', { gameId, playerNumber: currentPlayerNumber });
        return;
      }

      const { id, timestamp, moveNumber, ...rest } = aiMove;
      const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;

      const result = await gameEngine.makeMove(engineMove);
      if (!result.success) {
        logger.warn('Engine rejected AI move', {
          gameId,
          playerNumber: currentPlayerNumber,
          reason: result.error,
        });
        return;
      }

      const updatedState = gameEngine.getGameState();

      // Persist the AI move to the database using a dedicated AI user,
      // and, if the move ended the game, update the terminal status.
      const prisma = getDatabaseClient();
      if (prisma) {
        try {
          const aiUser = await getOrCreateAIUser();
          const lastMove = updatedState.moveHistory[updatedState.moveHistory.length - 1];

          await prisma.move.create({
            data: {
              gameId,
              playerId: aiUser.id,
              moveNumber: lastMove.moveNumber,
              position: JSON.stringify({ from: lastMove.from, to: lastMove.to }),
              moveType: lastMove.type as any,
              timestamp: lastMove.timestamp,
            },
          });

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
              where: { id: gameId },
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
            gameId,
            playerNumber: currentPlayerNumber,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      if (result.gameResult) {
        this.io.to(gameId).emit('game_over', {
          type: 'game_over',
          data: {
            gameId,
            gameState: updatedState,
            gameResult: result.gameResult,
          },
          timestamp: new Date().toISOString(),
        });

        logger.info('Game ended after AI move', {
          gameId,
          playerNumber: currentPlayerNumber,
          reason: result.gameResult.reason,
        });

        return;
      }

      this.io.to(gameId).emit('game_state', {
        type: 'game_update',
        data: {
          gameId,
          gameState: updatedState,
          validMoves: gameEngine.getValidMoves(updatedState.currentPlayer),
        },
        timestamp: new Date().toISOString(),
      });

      logger.info('AI move processed and applied', {
        gameId,
        playerNumber: currentPlayerNumber,
        moveType: engineMove.type,
      });
    } catch (error) {
      logger.error('Error during AI turn', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async handleChatMessage(socket: AuthenticatedSocket, data: any) {
    const { gameId, text } = data;

    if (!gameId || !text) {
      throw new Error('Missing gameId or text');
    }

    // Verify user is in the game room
    if (!socket.rooms.has(gameId)) {
      throw new Error('User not in game room');
    }

    // Broadcast chat message to all players in the game
    this.io.to(gameId).emit('chat_message', {
      sender: socket.username || 'Unknown',
      text,
      timestamp: new Date().toISOString(),
    });

    logger.info('Chat message sent', {
      userId: socket.userId,
      gameId,
      messageLength: text.length,
    });
  }

  private handleDisconnect(socket: AuthenticatedSocket) {
    logger.info('WebSocket disconnected', {
      userId: socket.userId,
      username: socket.username,
      socketId: socket.id,
    });

    // Remove from user socket mapping
    if (socket.userId) {
      this.userSockets.delete(socket.userId);
    }

    // Remove from game room tracking
    if (socket.gameId && this.gameRooms.has(socket.gameId)) {
      this.gameRooms.get(socket.gameId)!.delete(socket.id);
      if (this.gameRooms.get(socket.gameId)!.size === 0) {
        this.gameRooms.delete(socket.gameId);
      }

      // Notify others in the room
      socket.to(socket.gameId).emit('player_disconnected', {
        type: 'player_disconnected',
        data: {
          gameId: socket.gameId,
          player: {
            id: socket.userId!,
            username: socket.username!,
          },
        },
        timestamp: new Date().toISOString(),
      });
    }
  }

  // Public methods for external use
  public sendToUser(userId: string, event: string, data: any) {
    const socketId = this.userSockets.get(userId);
    if (socketId) {
      this.io.to(socketId).emit(event, data);
    }
  }

  public sendToGame(gameId: string, event: string, data: any) {
    this.io.to(gameId).emit(event, data);
  }

  public getConnectedUsers(): string[] {
    return Array.from(this.userSockets.keys());
  }

  public getGameRooms(): Map<string, Set<string>> {
    return this.gameRooms;
  }
}
