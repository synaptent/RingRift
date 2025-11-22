import { Server as SocketIOServer } from 'socket.io';
import { Server as HTTPServer } from 'http';
import { Socket } from 'socket.io';
import jwt from 'jsonwebtoken';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import { PlayerChoiceResponse } from '../../shared/types/game';
import { GameSessionManager } from '../game/GameSessionManager';

export interface AuthenticatedSocket extends Socket {
  userId?: string;
  username?: string;
  gameId?: string;
}

export class WebSocketServer {
  private io: SocketIOServer;
  private gameRooms: Map<string, Set<string>> = new Map();
  private userSockets: Map<string, string> = new Map();
  private sessionManager: GameSessionManager;

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

    this.sessionManager = new GameSessionManager(this.io, this.userSockets);

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

      // Handle player moves (geometry-based)
      socket.on('player_move', async (data: any) => {
        try {
          await this.handlePlayerMove(socket, data);
        } catch (error) {
          logger.error('Error handling player move:', error);
          socket.emit('error', { message: 'Invalid move' });
        }
      });

      // Handle canonical Move selection by id. This allows clients to:
      //   1. Fetch legal moves via game_state.validMoves (including advanced
      //      decision phases such as line_processing / territory_processing),
      //   2. Choose a Move.id on the client,
      //   3. Send { gameId, moveId } to have the backend resolve and apply
      //      the selected Move via GameEngine.makeMoveById.
      socket.on('player_move_by_id', async (data: { gameId: string; moveId: string }) => {
        try {
          await this.handlePlayerMoveById(socket, data);
        } catch (error) {
          logger.error('Error handling player move by id:', error);
          socket.emit('error', { message: 'Invalid move selection' });
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

          const session = this.sessionManager.getSession(gameId);
          if (!session) {
            throw new Error(`No active session found for gameId=${gameId}`);
          }

          session.getInteractionHandler().handleChoiceResponse(response);
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

  private async handleJoinGame(socket: AuthenticatedSocket, gameId: string) {
    // Wrap join logic in a lock to ensure consistent state when fetching/creating the engine
    await this.sessionManager.withGameLock(gameId, async () => {
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

      // Get or create game session
      const session = await this.sessionManager.getOrCreateSession(gameId);
      const gameState = session.getGameState();

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
          gameState,
          validMoves: isPlayer ? session.getValidMoves(gameState.currentPlayer) : [],
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
        gamePhase: gameState.currentPhase,
        currentPlayer: gameState.currentPlayer,
      });
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

    // Wrap move processing in a lock to prevent race conditions
    await this.sessionManager.withGameLock(gameId, async () => {
      const session = await this.sessionManager.getOrCreateSession(gameId);
      await session.handlePlayerMove(socket, move);
    });
  }

  /**
   * Handle a canonical Move selection identified by Move.id.
   */
  private async handlePlayerMoveById(
    socket: AuthenticatedSocket,
    data: { gameId: string; moveId: string }
  ) {
    const { gameId, moveId } = data;

    if (!socket.gameId || socket.gameId !== gameId) {
      throw new Error('Not in game room');
    }

    // Wrap move processing in a lock to prevent race conditions
    await this.sessionManager.withGameLock(gameId, async () => {
      const session = await this.sessionManager.getOrCreateSession(gameId);
      await session.handlePlayerMoveById(socket, moveId);
    });
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
