import { Server as SocketIOServer } from 'socket.io';
import { Server as HTTPServer } from 'http';
import { Socket } from 'socket.io';
import { ZodError } from 'zod';
import { getDatabaseClient } from '../database/connection';
import { getRedisClient } from '../cache/redis';
import { logger } from '../utils/logger';
import { PlayerChoiceResponse } from '../../shared/types/game';
import { GameSessionManager } from '../game/GameSessionManager';
import { config } from '../config';
import { WebSocketPayloadSchemas } from '../../shared/validation/websocketSchemas';
import {
  WebSocketErrorCode,
  WebSocketErrorPayload,
  ClientToServerEvents,
  ServerToClientEvents,
} from '../../shared/types/websocket';
import { webSocketConnectionsGauge } from '../utils/rulesParityMetrics';
import { getMetricsService } from '../services/MetricsService';
import { verifyToken, validateUser } from '../middleware/auth';
import {
  type PlayerConnectionState,
  markConnected,
  markDisconnectedExpired,
  markDisconnectedPendingReconnect,
} from '../../shared/stateMachines/connection';

export interface AuthenticatedSocket extends Socket<ClientToServerEvents, ServerToClientEvents> {
  userId?: string;
  username?: string;
  gameId?: string;
}

const CHAT_RATE_LIMIT_WINDOW_SECONDS = 10;
const CHAT_RATE_LIMIT_MAX_MESSAGES = 20;

type InMemoryChatCounter = {
  count: number;
  resetAt: number;
};

const inMemoryChatCounters = new Map<string, InMemoryChatCounter>();

async function checkChatRateLimit(socket: AuthenticatedSocket, gameId: string): Promise<boolean> {
  const userKey = socket.userId ?? socket.id;
  const key = `chat:${gameId}:${userKey}`;
  const windowSeconds = CHAT_RATE_LIMIT_WINDOW_SECONDS;
  const maxMessages = CHAT_RATE_LIMIT_MAX_MESSAGES;

  try {
    const redis = getRedisClient();

    if (redis) {
      try {
        const current = await redis.incr(key);
        if (current === 1) {
          await redis.expire(key, windowSeconds);
        }

        if (current > maxMessages) {
          logger.warn('Chat rate limit exceeded (redis)', {
            userId: socket.userId,
            gameId,
            key,
            count: current,
          });
          return false;
        }

        return true;
      } catch (error) {
        logger.warn('Chat rate limiter Redis error, falling back to in-memory', {
          userId: socket.userId,
          gameId,
          key,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const now = Date.now();
    const windowMs = windowSeconds * 1000;

    const existing = inMemoryChatCounters.get(key);
    if (!existing || existing.resetAt <= now) {
      inMemoryChatCounters.set(key, { count: 1, resetAt: now + windowMs });
      return true;
    }

    if (existing.count >= maxMessages) {
      logger.warn('Chat rate limit exceeded (memory)', {
        userId: socket.userId,
        gameId,
        key,
        count: existing.count + 1,
      });
      return false;
    }

    existing.count += 1;
    return true;
  } catch (error) {
    logger.warn('Chat rate limiter failure, allowing chat message', {
      userId: socket.userId,
      gameId,
      key,
      error: error instanceof Error ? error.message : String(error),
    });
    return true;
  }
}

// Default reconnection window in milliseconds - players have this long to reconnect
// before their choices are cleared and they're fully removed from the session
const RECONNECTION_TIMEOUT_MS = 30_000;

export class WebSocketServer {
  private io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>;
  private gameRooms: Map<string, Set<string>> = new Map();
  private userSockets: Map<string, string> = new Map();
  private sessionManager: GameSessionManager;
  private static LOBBY_ROOM = 'lobby';

  // Track users who have disconnected but might reconnect
  // Key: `${gameId}:${userId}`, Value: { timeout, playerNumber, gameId, userId }
  private pendingReconnections: Map<
    string,
    {
      timeout: NodeJS.Timeout;
      playerNumber: number;
      gameId: string;
      userId: string;
    }
  > = new Map();

  /**
   * Diagnostic view of per-player connection state for each game. This is
   * keyed by `${gameId}:${userId}` and is primarily used for tests and
   * incident debugging rather than gameplay logic.
   */
  private readonly playerConnectionStates = new Map<string, PlayerConnectionState>();

  constructor(httpServer: HTTPServer) {
    // Align WebSocket CORS with the HTTP API CORS configuration. Prefer an
    // explicit CLIENT_URL, then fall back to CORS_ORIGIN (used by Express),
    // then the first entry in ALLOWED_ORIGINS, and finally the Vite dev
    // origin http://localhost:5173. This avoids local dev failures where the
    // Socket.IO layer only allowed http://localhost:3000.
    const allowedOrigin = config.server.websocketOrigin;

    this.io = new SocketIOServer<ClientToServerEvents, ServerToClientEvents>(httpServer, {
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
        const rawToken =
          (socket.handshake.auth && (socket.handshake.auth as any).token) ||
          socket.handshake.query.token;

        if (!rawToken || typeof rawToken !== 'string') {
          const payload: WebSocketErrorPayload = {
            type: 'error',
            code: 'ACCESS_DENIED',
            message: 'Authentication token required',
          };
          socket.emit('error', payload);
          return next(new Error('Authentication token required'));
        }

        let decoded;
        try {
          decoded = verifyToken(rawToken);
        } catch (err) {
          logger.warn('WebSocket JWT verification failed', {
            error: err instanceof Error ? err.message : String(err),
            socketId: socket.id,
          });

          const payload: WebSocketErrorPayload = {
            type: 'error',
            code: 'ACCESS_DENIED',
            message: 'Authentication failed',
          };
          socket.emit('error', payload);

          return next(new Error('Authentication failed'));
        }

        try {
          const user = await validateUser(decoded.userId, decoded.tokenVersion);

          socket.userId = user.id;
          socket.username = user.username;

          logger.info('WebSocket authenticated', {
            userId: user.id,
            username: user.username,
            socketId: socket.id,
            connectionId: socket.id,
          });

          next();
        } catch (err) {
          logger.warn('WebSocket auth user validation failed', {
            error: err instanceof Error ? err.message : String(err),
            socketId: socket.id,
            userId: decoded.userId,
          });

          const payload: WebSocketErrorPayload = {
            type: 'error',
            code: 'ACCESS_DENIED',
            message: 'Authentication failed',
          };
          socket.emit('error', payload);

          return next(new Error('Authentication failed'));
        }
      } catch (error) {
        logger.error('WebSocket authentication middleware threw unexpectedly', {
          error: error instanceof Error ? error.message : String(error),
          socketId: socket.id,
        });
        const payload: WebSocketErrorPayload = {
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'Authentication failed',
        };
        socket.emit('error', payload);
        next(new Error('Authentication failed'));
      }
    });
  }

  private setupEventHandlers() {
    this.io.on('connection', (socket: AuthenticatedSocket) => {
      // Track active WebSocket connections for Prometheus metrics.
      // We update both the legacy gauge and the MetricsService for consistency.
      webSocketConnectionsGauge.inc();
      getMetricsService().incWebSocketConnections();

      logger.info('WebSocket connected', {
        userId: socket.userId,
        username: socket.username,
        socketId: socket.id,
        connectionId: socket.id,
      });

      // Store user socket mapping
      if (socket.userId) {
        this.userSockets.set(socket.userId, socket.id);
      }

      // Join game room
      socket.on('join_game', async (data: unknown) => {
        try {
          const { gameId } = WebSocketPayloadSchemas.join_game.parse(data);
          await this.handleJoinGame(socket, gameId);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'join_game', error);
            return;
          }

          const message = error instanceof Error ? error.message : String(error);

          if (message === 'Game not found') {
            this.emitError(socket, 'GAME_NOT_FOUND', 'Game not found', 'join_game');
          } else if (message === 'Access denied') {
            this.emitError(
              socket,
              'ACCESS_DENIED',
              'You are not allowed to join this game',
              'join_game'
            );
          } else {
            logger.error('Error joining game', {
              gameId: socket.gameId,
              socketId: socket.id,
              userId: socket.userId,
              error: error instanceof Error ? error.message : String(error),
            });
            this.emitError(socket, 'INTERNAL_ERROR', 'Failed to join game', 'join_game');
          }
        }
      });

      // Leave game room
      socket.on('leave_game', async (data: unknown) => {
        try {
          const { gameId } = WebSocketPayloadSchemas.leave_game.parse(data);
          await this.handleLeaveGame(socket, gameId);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'leave_game', error);
            return;
          }
          logger.error('Error leaving game', {
            gameId: socket.gameId,
            socketId: socket.id,
            userId: socket.userId,
            error: error instanceof Error ? error.message : String(error),
          });
          this.emitError(socket, 'INTERNAL_ERROR', 'Failed to leave game', 'leave_game');
        }
      });

      // Lobby subscription handlers
      socket.on('lobby:subscribe', () => {
        socket.join(WebSocketServer.LOBBY_ROOM);
        logger.info('Client subscribed to lobby updates', {
          socketId: socket.id,
          userId: socket.userId,
        });
      });

      socket.on('lobby:unsubscribe', () => {
        socket.leave(WebSocketServer.LOBBY_ROOM);
        logger.info('Client unsubscribed from lobby updates', {
          socketId: socket.id,
          userId: socket.userId,
        });
      });

      // Handle player moves (geometry-based)
      socket.on('player_move', async (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas.player_move.parse(data);
          await this.handlePlayerMove(socket, payload);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'player_move', error);
            return;
          }

          const message = error instanceof Error ? error.message : String(error);

          if (message === 'Database not available') {
            logger.error('Error handling player move (database not available)', {
              gameId: socket.gameId,
              socketId: socket.id,
              userId: socket.userId,
              error: error instanceof Error ? error.message : String(error),
            });
            this.emitError(socket, 'INTERNAL_ERROR', 'Unable to process move', 'player_move');
          } else if (message === 'Game not found') {
            this.emitError(socket, 'GAME_NOT_FOUND', 'Game not found', 'player_move');
          } else if (message === 'Game is not active') {
            this.emitError(socket, 'INTERNAL_ERROR', 'Game is not active', 'player_move');
          } else if (
            message === 'Spectators cannot make moves' ||
            message === 'Current socket user is not a player in this game' ||
            message === 'Not in game room'
          ) {
            this.emitError(
              socket,
              'ACCESS_DENIED',
              'You are not allowed to make moves in this game',
              'player_move'
            );
          } else if (
            message === 'Invalid move position payload' ||
            message === 'Move destination is required'
          ) {
            this.emitError(socket, 'INVALID_PAYLOAD', 'Invalid move payload', 'player_move');
          } else {
            // Default: treat as a rules-engine rejection of an illegal move.
            logger.warn('Engine rejected move', {
              socketId: socket.id,
              userId: socket.userId,
              gameId: socket.gameId,
              message,
            });
            this.emitError(
              socket,
              'MOVE_REJECTED',
              'Move was not valid in the current game state',
              'player_move'
            );
          }
        }
      });

      // Handle canonical Move selection by id. This allows clients to:
      //   1. Fetch legal moves via game_state.validMoves (including advanced
      //      decision phases such as line_processing / territory_processing),
      //   2. Choose a Move.id on the client,
      //   3. Send { gameId, moveId } to have the backend resolve and apply
      //      the selected Move via GameEngine.makeMoveById.
      socket.on('player_move_by_id', async (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas.player_move_by_id.parse(data);
          await this.handlePlayerMoveById(socket, payload);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'player_move_by_id', error);
            return;
          }

          const message = error instanceof Error ? error.message : String(error);

          if (message === 'Database not available') {
            logger.error('Error handling player move by id (database not available)', {
              gameId: socket.gameId,
              socketId: socket.id,
              userId: socket.userId,
              error: error instanceof Error ? error.message : String(error),
            });
            this.emitError(
              socket,
              'INTERNAL_ERROR',
              'Unable to process move selection',
              'player_move_by_id'
            );
          } else if (message === 'Game not found') {
            this.emitError(socket, 'GAME_NOT_FOUND', 'Game not found', 'player_move_by_id');
          } else if (message === 'Game is not active') {
            this.emitError(socket, 'INTERNAL_ERROR', 'Game is not active', 'player_move_by_id');
          } else if (
            message === 'Spectators cannot make moves' ||
            message === 'Current socket user is not a player in this game' ||
            message === 'Not in game room'
          ) {
            this.emitError(
              socket,
              'ACCESS_DENIED',
              'You are not allowed to make moves in this game',
              'player_move_by_id'
            );
          } else {
            // Default: treat as a rules-engine rejection of an illegal move selection.
            logger.warn('Engine rejected move by id', {
              socketId: socket.id,
              userId: socket.userId,
              gameId: socket.gameId,
              message,
            });
            this.emitError(
              socket,
              'MOVE_REJECTED',
              'Move selection was not valid in the current game state',
              'player_move_by_id'
            );
          }
        }
      });

      // Handle chat messages
      socket.on('chat_message', async (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas.chat_message.parse(data);
          await this.handleChatMessage(socket, payload);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'chat_message', error);
            return;
          }

          const message = error instanceof Error ? error.message : String(error);

          if (message === 'User not in game room') {
            // Authorization invariant: only sockets that have successfully joined
            // the game room (via join_game) may send chat messages for that game.
            this.emitError(
              socket,
              'ACCESS_DENIED',
              'You are not allowed to chat in this game',
              'chat_message'
            );
          } else {
            logger.error('Error handling chat message', {
              socketId: socket.id,
              userId: socket.userId,
              gameId: socket.gameId,
              error: message,
            });
            this.emitError(socket, 'INVALID_PAYLOAD', 'Invalid chat message', 'chat_message');
          }
        }
      });

      // Handle player choice responses
      socket.on('player_choice_response', (data: unknown) => {
        try {
          const response = WebSocketPayloadSchemas.player_choice_response.parse(
            data
          ) as PlayerChoiceResponse<unknown>;

          const gameId = socket.gameId;
          if (!gameId) {
            throw new Error('player_choice_response received without an active gameId');
          }

          const session = this.sessionManager.getSession(gameId);
          if (!session) {
            throw new Error(`No active session found for gameId=${gameId}`);
          }

          // Game-level authorization: ensure that the responding socket
          // belongs to the human player associated with response.playerNumber.
          const gameState = session.getGameState();

          // Spectator enforcement: strictly reject choice responses from spectators
          if (socket.userId && gameState.spectators.includes(socket.userId)) {
            throw new Error('Spectators cannot respond to player choices');
          }

          const player = gameState.players.find((p) => p.playerNumber === response.playerNumber);

          if (!player || player.id !== socket.userId) {
            throw new Error('Current socket user is not the player for this choice');
          }

          session.getInteractionHandler().handleChoiceResponse(response);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'player_choice_response', error);
            return;
          }

          const message = error instanceof Error ? error.message : String(error);

          if (message === 'Spectators cannot respond to player choices') {
            this.emitError(
              socket,
              'ACCESS_DENIED',
              'Spectators cannot respond to player choices',
              'player_choice_response'
            );
          } else if (
            message.startsWith('playerNumber mismatch for choice') ||
            message.startsWith('Invalid selectedOption for choice')
          ) {
            this.emitError(
              socket,
              'CHOICE_REJECTED',
              'Choice response was not valid for the current game state',
              'player_choice_response'
            );
          } else if (message === 'Current socket user is not the player for this choice') {
            this.emitError(
              socket,
              'CHOICE_REJECTED',
              'You are not allowed to respond to this choice',
              'player_choice_response'
            );
          } else if (message.includes('No active session found')) {
            this.emitError(socket, 'GAME_NOT_FOUND', 'Game not found', 'player_choice_response');
          } else {
            logger.error('Error handling player_choice_response', {
              socketId: socket.id,
              userId: socket.userId,
              gameId: socket.gameId,
              error: error instanceof Error ? error.message : String(error),
            });
            this.emitError(
              socket,
              'INTERNAL_ERROR',
              'Invalid choice response',
              'player_choice_response'
            );
          }
        }
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        this.handleDisconnect(socket);
      });
    });
  }

  private emitError(
    socket: AuthenticatedSocket,
    code: WebSocketErrorCode,
    message: string,
    event?: string
  ): void {
    logger.warn('WebSocket error', {
      code,
      event,
      socketId: socket.id,
      userId: socket.userId,
      gameId: socket.gameId,
    });

    const payload: WebSocketErrorPayload = {
      type: 'error',
      code,
      message,
      ...(event ? { event } : {}),
    };

    socket.emit('error', payload);
  }

  private handleWebSocketValidationError(
    socket: AuthenticatedSocket,
    eventName: string,
    error: unknown
  ) {
    logger.warn('Rejected WebSocket payload due to validation error', {
      eventName,
      socketId: socket.id,
      userId: socket.userId,
      gameId: socket.gameId,
      error: error instanceof Error ? { name: error.name, message: error.message } : error,
    });

    this.emitError(socket, 'INVALID_PAYLOAD', 'Invalid payload', eventName);
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

      // Check if this is a reconnection - clear any pending disconnect timeout
      const reconnectionKey = `${gameId}:${socket.userId}`;
      const pendingReconnection = this.pendingReconnections.get(reconnectionKey);
      const isReconnection = !!pendingReconnection;

      if (pendingReconnection) {
        clearTimeout(pendingReconnection.timeout);
        this.pendingReconnections.delete(reconnectionKey);

        // Record successful reconnection metric
        getMetricsService().recordWebsocketReconnection('success');

        logger.info('Player reconnected within window', {
          userId: socket.userId,
          gameId,
          socketId: socket.id,
          playerNumber: pendingReconnection.playerNumber,
        });
      }

      // Get or create game session
      const session = await this.sessionManager.getOrCreateSession(gameId);
      const gameState = session.getGameState();

      // Update per-player connection state diagnostics for this game.
      if (socket.userId) {
        const key = `${gameId}:${socket.userId}`;
        const previous = this.playerConnectionStates.get(key);
        const player = gameState.players.find((p) => p.id === socket.userId);
        const playerNumber = player?.playerNumber;
        const nextState = markConnected(gameId, socket.userId, playerNumber, previous);
        this.playerConnectionStates.set(key, nextState);
      }

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

      // Notify others in the room - use appropriate event based on reconnection status
      if (isReconnection && pendingReconnection) {
        socket.to(gameId).emit('player_reconnected', {
          type: 'player_reconnected',
          data: {
            gameId,
            player: {
              id: socket.userId!,
              username: socket.username!,
            },
            playerNumber: pendingReconnection.playerNumber,
          },
          timestamp: new Date().toISOString(),
        });
      } else {
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
      }

      logger.info('Player joined game room', {
        userId: socket.userId,
        gameId,
        socketId: socket.id,
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
      socketId: socket.id,
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

    // Enforce a per-user, per-game chat rate limit to mitigate spam and simple
    // abuse without impacting normal play.
    const allowed = await checkChatRateLimit(socket, gameId);
    if (!allowed) {
      this.emitError(socket, 'RATE_LIMITED', 'Chat rate limit exceeded', 'chat_message');
      return;
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
      socketId: socket.id,
      messageLength: text.length,
    });
  }

  private handleDisconnect(socket: AuthenticatedSocket) {
    // Decrement active connection gauge. This mirrors the single increment that
    // occurs in the 'connection' handler and should be called exactly once per
    // socket lifecycle.
    webSocketConnectionsGauge.dec();
    getMetricsService().decWebSocketConnections();

    logger.info('WebSocket disconnected', {
      userId: socket.userId,
      username: socket.username,
      socketId: socket.id,
      connectionId: socket.id,
      gameId: socket.gameId,
    });

    // Remove from user socket mapping immediately
    // (will be re-added if they reconnect)
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

      // Set up reconnection window for players
      // If they don't reconnect within the window, clear their stale choices
      const gameId = socket.gameId;
      const userId = socket.userId;

      if (userId) {
        const session = this.sessionManager.getSession(gameId);
        if (session) {
          const gameState = session.getGameState();
          const player = gameState.players.find(
            (p: { id: string; playerNumber: number }) => p.id === userId
          );

          // Only set up reconnection window for actual players, not spectators
          if (player && !gameState.spectators.includes(userId)) {
            const reconnectionKey = `${gameId}:${userId}`;

            // Clear any existing timeout for this user/game combo
            const existing = this.pendingReconnections.get(reconnectionKey);
            if (existing) {
              clearTimeout(existing.timeout);
            }

            const timeout = setTimeout(() => {
              this.handleReconnectionTimeout(gameId, userId, player.playerNumber);
            }, RECONNECTION_TIMEOUT_MS);

            this.pendingReconnections.set(reconnectionKey, {
              timeout,
              playerNumber: player.playerNumber,
              gameId,
              userId,
            });

            // Update connection state diagnostics to reflect a pending reconnect window.
            const key = `${gameId}:${userId}`;
            const previous = this.playerConnectionStates.get(key);
            const nextState = markDisconnectedPendingReconnect(
              previous,
              gameId,
              userId,
              player.playerNumber,
              RECONNECTION_TIMEOUT_MS
            );
            this.playerConnectionStates.set(key, nextState);

            logger.info('Reconnection window started', {
              userId,
              gameId,
              playerNumber: player.playerNumber,
              timeoutMs: RECONNECTION_TIMEOUT_MS,
            });
          }
        }
      }
    }
  }

  /**
   * Called when a player's reconnection window expires.
   * Clears any stale pending choices for that player.
   */
  private handleReconnectionTimeout(gameId: string, userId: string, playerNumber: number): void {
    const reconnectionKey = `${gameId}:${userId}`;
    this.pendingReconnections.delete(reconnectionKey);

    // Record timeout for reconnection metric
    getMetricsService().recordWebsocketReconnection('timeout');

    logger.info('Reconnection window expired, clearing stale choices', {
      userId,
      gameId,
      playerNumber,
    });

    // Update connection state diagnostics to reflect an expired reconnection window.
    const key = `${gameId}:${userId}`;
    const previous = this.playerConnectionStates.get(key);
    const nextState = markDisconnectedExpired(previous, gameId, userId, playerNumber);
    this.playerConnectionStates.set(key, nextState);

    const session = this.sessionManager.getSession(gameId);
    if (session) {
      // Cancel all pending choices for this player
      session.getInteractionHandler().cancelAllChoicesForPlayer(playerNumber);
    }
  }

  // Public methods for external use
  public sendToUser(userId: string, event: keyof ServerToClientEvents, data: any) {
    const socketId = this.userSockets.get(userId);
    if (socketId) {
      this.io.to(socketId).emit(event, data);
    }
  }

  /**
   * Terminate all WebSocket connections for a given user.
   * Used when a user's account is deleted or their sessions need to be forcibly ended.
   * The connection is closed with a custom close code (4001) and reason message.
   *
   * @param userId - The ID of the user whose sessions should be terminated
   * @param reason - Optional reason string sent to the client (default: 'Session terminated')
   * @returns The number of connections that were terminated
   */
  public terminateUserSessions(userId: string, reason: string = 'Session terminated'): number {
    const socketId = this.userSockets.get(userId);
    if (!socketId) {
      logger.info('No active WebSocket connections found for user', { userId, reason });
      return 0;
    }

    const socket = this.io.sockets.sockets.get(socketId) as AuthenticatedSocket | undefined;
    if (!socket) {
      // Socket ID was tracked but socket is no longer in the server's collection
      // Clean up the stale mapping
      this.userSockets.delete(userId);
      logger.info('Stale socket mapping cleaned up for user', { userId, socketId, reason });
      return 0;
    }

    // Emit an error event to inform the client before disconnecting
    const payload: WebSocketErrorPayload = {
      type: 'error',
      code: 'ACCESS_DENIED',
      message: reason,
    };
    socket.emit('error', payload);

    // Disconnect the socket - the second parameter (true) forces immediate disconnection
    socket.disconnect(true);

    // Clean up tracking maps
    this.userSockets.delete(userId);

    // Clean up any pending reconnection timeouts for this user
    for (const [key, pending] of this.pendingReconnections.entries()) {
      if (pending.userId === userId) {
        clearTimeout(pending.timeout);
        this.pendingReconnections.delete(key);
        logger.info('Cleared pending reconnection for terminated user', {
          userId,
          gameId: pending.gameId,
        });
      }
    }

    logger.info('WebSocket session terminated', {
      userId,
      socketId,
      reason,
      terminatedCount: 1,
    });

    return 1;
  }

  public sendToGame(gameId: string, event: keyof ServerToClientEvents, data: any) {
    this.io.to(gameId).emit(event, data);
  }

  public getConnectedUsers(): string[] {
    return Array.from(this.userSockets.keys());
  }

  public getGameRooms(): Map<string, Set<string>> {
    return this.gameRooms;
  }

  /**
   * Expose a snapshot of the last known PlayerConnectionState for a given
   * game/user pair. This is primarily used in tests.
   */
  public getPlayerConnectionStateSnapshotForTesting(
    gameId: string,
    userId: string
  ): PlayerConnectionState | undefined {
    const key = `${gameId}:${userId}`;
    return this.playerConnectionStates.get(key);
  }

  /**
   * Aggregate in-memory diagnostics for a given game, combining the
   * GameSession-derived projections (when available) with the current
   * connection state machine snapshots. This method is intentionally
   * loosely typed and uses the public testing/diagnostics accessors on
   * GameSession so that HTTP routes and incident tooling can consume a
   * compact view without tightening coupling.
   */
  public getGameDiagnosticsForGame(gameId: string): {
    sessionStatus: any | null;
    lastAIRequestState: any | null;
    aiDiagnostics: any | null;
    connections: Record<string, PlayerConnectionState>;
    hasInMemorySession: boolean;
  } {
    const session = this.sessionManager.getSession(gameId) as any | undefined;

    const connections: Record<string, PlayerConnectionState> = {};
    const prefix = `${gameId}:`;
    for (const [key, state] of this.playerConnectionStates.entries()) {
      if (key.startsWith(prefix)) {
        const userId = key.slice(prefix.length);
        connections[userId] = state;
      }
    }

    if (!session) {
      return {
        sessionStatus: null,
        lastAIRequestState: null,
        aiDiagnostics: null,
        connections,
        hasInMemorySession: false,
      };
    }

    const sessionStatus =
      typeof session.getSessionStatusSnapshot === 'function'
        ? session.getSessionStatusSnapshot()
        : null;
    const lastAIRequestState =
      typeof session.getLastAIRequestStateForTesting === 'function'
        ? session.getLastAIRequestStateForTesting()
        : null;
    const aiDiagnostics =
      typeof session.getAIDiagnosticsSnapshotForTesting === 'function'
        ? session.getAIDiagnosticsSnapshotForTesting()
        : null;

    return {
      sessionStatus,
      lastAIRequestState,
      aiDiagnostics,
      connections,
      hasInMemorySession: true,
    };
  }

  // Lobby broadcast methods
  public broadcastLobbyEvent(event: keyof ServerToClientEvents, data: any) {
    this.io.to(WebSocketServer.LOBBY_ROOM).emit(event, data);
    logger.debug('Broadcast lobby event', { event, lobbyRoom: WebSocketServer.LOBBY_ROOM });
  }
}
