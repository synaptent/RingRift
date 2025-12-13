import { Server as SocketIOServer } from 'socket.io';
import { Server as HTTPServer } from 'http';
import { Socket } from 'socket.io';
import { ZodError } from 'zod';
import { getDatabaseClient } from '../database/connection';
import { getRedisClient } from '../cache/redis';
import { logger } from '../utils/logger';
import { PlayerChoiceResponse } from '../../shared/types/game';
import { GameSessionManager } from '../game/GameSessionManager';
import type { RulesResult } from '../game/RulesBackendFacade';
import { config } from '../config';
import {
  WebSocketPayloadSchemas,
  PlayerMovePayload,
  ChatMessagePayload,
  RematchRequestPayload,
  RematchResponsePayload,
  MatchmakingJoinPayload,
} from '../../shared/validation/websocketSchemas';
import {
  WebSocketErrorCode,
  WebSocketErrorPayload,
  ClientToServerEvents,
  ServerToClientEvents,
  DiagnosticPongPayload,
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
import { getChatPersistenceService } from '../services/ChatPersistenceService';
import { getRematchService } from '../services/RematchService';
import { MatchmakingService } from '../services/MatchmakingService';

/**
 * Extract only the keys from ServerToClientEvents that have defined (non-optional) handlers.
 * This ensures generic emit helpers work correctly with TypeScript's Parameters<> utility.
 */
type DefinedServerEvent = {
  [K in keyof ServerToClientEvents]-?: ServerToClientEvents[K] extends (...args: infer _A) => void
    ? K
    : never;
}[keyof ServerToClientEvents];

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

export class WebSocketServer {
  private io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>;
  private gameRooms: Map<string, Set<string>> = new Map();
  private userSockets: Map<string, string> = new Map();
  private sessionManager: GameSessionManager;
  private matchmakingService: MatchmakingService;
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
      // Align Socket.IO ping timeout with the configured reconnection window
      // so clients that lose connectivity are detected within the same
      // timeframe used by our server-side reconnection grace period.
      pingTimeout: config.server.wsReconnectionTimeoutMs,
    });

    this.sessionManager = new GameSessionManager(this.io, this.userSockets);
    this.matchmakingService = new MatchmakingService(this);

    this.setupMiddleware();
    this.setupEventHandlers();
  }

  private setupMiddleware() {
    // Authentication middleware
    this.io.use(async (socket: AuthenticatedSocket, next) => {
      try {
        const rawToken =
          (socket.handshake.auth && (socket.handshake.auth as Record<string, unknown>).token) ||
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

      // Matchmaking handlers
      socket.on('matchmaking:join', async (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas['matchmaking:join'].parse(data);

          if (!socket.userId) {
            this.emitError(
              socket,
              'ACCESS_DENIED',
              'Authentication required to join matchmaking',
              'matchmaking:join'
            );
            return;
          }

          // Get user rating from database with timeout protection.
          // Falls back to default rating (1200) if DB is slow or unavailable.
          const DB_QUERY_TIMEOUT_MS = 5000;
          const prisma = getDatabaseClient();
          let rating = 1200; // Default rating

          if (prisma) {
            try {
              const timeoutPromise = new Promise<null>((resolve) =>
                setTimeout(() => resolve(null), DB_QUERY_TIMEOUT_MS)
              );
              const queryPromise = prisma.user.findUnique({
                where: { id: socket.userId },
                select: { rating: true },
              });

              const user = await Promise.race([queryPromise, timeoutPromise]);
              if (user?.rating !== undefined) {
                rating = user.rating;
              } else if (user === null) {
                logger.warn('Matchmaking DB query timed out, using default rating', {
                  userId: socket.userId,
                  timeoutMs: DB_QUERY_TIMEOUT_MS,
                });
              }
            } catch (dbError) {
              logger.warn('Matchmaking DB query failed, using default rating', {
                userId: socket.userId,
                error: dbError instanceof Error ? dbError.message : String(dbError),
              });
            }
          }

          this.matchmakingService.addToQueue(socket.userId, socket.id, payload.preferences, rating);

          logger.info('User joined matchmaking queue', {
            userId: socket.userId,
            preferences: payload.preferences,
          });
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'matchmaking:join', error);
            return;
          }

          logger.error('Failed to join matchmaking queue', {
            userId: socket.userId,
            error: error instanceof Error ? error.message : String(error),
          });
          this.emitError(
            socket,
            'INTERNAL_ERROR',
            'Failed to join matchmaking queue',
            'matchmaking:join'
          );
        }
      });

      socket.on('matchmaking:leave', () => {
        if (socket.userId) {
          this.matchmakingService.removeFromQueue(socket.userId);
          logger.info('User left matchmaking queue', { userId: socket.userId });
        }
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
            getMetricsService().recordMoveRejected('db_unavailable');
            this.emitError(socket, 'INTERNAL_ERROR', 'Unable to process move', 'player_move');
          } else if (message === 'Game not found') {
            getMetricsService().recordMoveRejected('game_not_found');
            this.emitError(socket, 'GAME_NOT_FOUND', 'Game not found', 'player_move');
          } else if (message === 'Game is not active') {
            getMetricsService().recordMoveRejected('game_not_active');
            this.emitError(socket, 'INTERNAL_ERROR', 'Game is not active', 'player_move');
          } else if (
            message === 'Spectators cannot make moves' ||
            message === 'Current socket user is not a player in this game' ||
            message === 'Not in game room'
          ) {
            getMetricsService().recordMoveRejected('authz');
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
            getMetricsService().recordMoveRejected('invalid_payload');
            this.emitError(socket, 'INVALID_PAYLOAD', 'Invalid move payload', 'player_move');
          } else {
            // Default: treat as a rules-engine rejection of an illegal move.
            logger.warn('Engine rejected move', {
              socketId: socket.id,
              userId: socket.userId,
              gameId: socket.gameId,
              message,
            });
            getMetricsService().recordMoveRejected('rules_invalid');
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
            getMetricsService().recordMoveRejected('db_unavailable');
            this.emitError(
              socket,
              'INTERNAL_ERROR',
              'Unable to process move selection',
              'player_move_by_id'
            );
          } else if (message === 'Game not found') {
            getMetricsService().recordMoveRejected('game_not_found');
            this.emitError(socket, 'GAME_NOT_FOUND', 'Game not found', 'player_move_by_id');
          } else if (message === 'Game is not active') {
            getMetricsService().recordMoveRejected('game_not_active');
            this.emitError(socket, 'INTERNAL_ERROR', 'Game is not active', 'player_move_by_id');
          } else if (
            message === 'Spectators cannot make moves' ||
            message === 'Current socket user is not a player in this game' ||
            message === 'Not in game room'
          ) {
            getMetricsService().recordMoveRejected('authz');
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
            getMetricsService().recordMoveRejected('rules_invalid');
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

      // Lightweight diagnostic ping/pong channel used by load tests.
      // This is intentionally transport-only and does not touch game
      // state, the rules engine, or the database.
      socket.on('diagnostic:ping', (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas['diagnostic:ping'].parse(data);

          // Echo the payload back with a server-side timestamp so clients
          // (including k6) can compute round-trip latency.
          const pongPayload: DiagnosticPongPayload = {
            timestamp: payload.timestamp,
            serverTimestamp: new Date().toISOString(),
            ...(payload.vu !== undefined && { vu: payload.vu }),
            ...(typeof payload.sequence === 'number' && { sequence: payload.sequence }),
          };

          socket.emit('diagnostic:pong', pongPayload);
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'diagnostic:ping', error);
            return;
          }

          logger.warn('Error handling diagnostic:ping', {
            socketId: socket.id,
            userId: socket.userId,
            gameId: socket.gameId,
            error: error instanceof Error ? error.message : String(error),
          });

          this.emitError(
            socket,
            'INVALID_PAYLOAD',
            'Invalid diagnostic ping payload',
            'diagnostic:ping'
          );
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

      // Handle rematch requests
      socket.on('rematch_request', async (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas.rematch_request.parse(data);

          if (!socket.userId) {
            this.emitError(socket, 'ACCESS_DENIED', 'User not authenticated', 'rematch_request');
            return;
          }

          const rematchService = getRematchService();
          const result = await rematchService.createRematchRequest(payload.gameId, socket.userId);

          if (!result.success || !result.request) {
            this.emitError(
              socket,
              'INVALID_PAYLOAD',
              result.error || 'Failed to create rematch request',
              'rematch_request'
            );
            return;
          }

          // Broadcast to all players in the game room
          this.io.to(payload.gameId).emit('rematch_requested', {
            id: result.request.id,
            gameId: result.request.gameId,
            requesterId: result.request.requesterId,
            requesterUsername: result.request.requesterUsername,
            expiresAt: result.request.expiresAt.toISOString(),
          });

          logger.info('Rematch request broadcast', {
            requestId: result.request.id,
            gameId: payload.gameId,
            requesterId: socket.userId,
          });
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'rematch_request', error);
            return;
          }

          logger.error('Error handling rematch_request', {
            socketId: socket.id,
            userId: socket.userId,
            error: error instanceof Error ? error.message : String(error),
          });
          this.emitError(
            socket,
            'INTERNAL_ERROR',
            'Failed to process rematch request',
            'rematch_request'
          );
        }
      });

      // Handle rematch responses (accept/decline)
      socket.on('rematch_respond', async (data: unknown) => {
        try {
          const payload = WebSocketPayloadSchemas.rematch_respond.parse(data);

          if (!socket.userId) {
            this.emitError(socket, 'ACCESS_DENIED', 'User not authenticated', 'rematch_respond');
            return;
          }

          const rematchService = getRematchService();

          if (payload.accept) {
            // Accept rematch - creates a new game
            const result = await rematchService.acceptRematch(
              payload.requestId,
              socket.userId,
              async (originalGameId: string) => {
                // Create a new game based on the original game settings
                const prisma = getDatabaseClient();
                if (!prisma) {
                  throw new Error('Database not available');
                }

                const originalGame = await prisma.game.findUnique({
                  where: { id: originalGameId },
                  select: {
                    boardType: true,
                    maxPlayers: true,
                    timeControl: true,
                    isRated: true,
                    allowSpectators: true,
                    player1Id: true,
                    player2Id: true,
                    player3Id: true,
                    player4Id: true,
                  },
                });

                if (!originalGame) {
                  throw new Error('Original game not found');
                }

                // Create new game with swapped player positions (for fairness)
                const newGame = await prisma.game.create({
                  data: {
                    boardType: originalGame.boardType,
                    maxPlayers: originalGame.maxPlayers,
                    timeControl: originalGame.timeControl ?? {
                      initialTime: 300000,
                      increment: 5000,
                    },
                    isRated: originalGame.isRated,
                    allowSpectators: originalGame.allowSpectators,
                    status: 'waiting',
                    player1Id: originalGame.player2Id, // Swap first two players
                    player2Id: originalGame.player1Id,
                    player3Id: originalGame.player3Id,
                    player4Id: originalGame.player4Id,
                  },
                });

                return newGame.id;
              }
            );

            if (!result.success || !result.request) {
              this.emitError(
                socket,
                'INVALID_PAYLOAD',
                result.error || 'Failed to accept rematch',
                'rematch_respond'
              );
              return;
            }

            // Broadcast response to all players in the original game room
            this.io.to(result.request.gameId).emit('rematch_response', {
              requestId: payload.requestId,
              gameId: result.request.gameId,
              status: 'accepted' as const,
              ...(result.newGameId ? { newGameId: result.newGameId } : {}),
            });

            logger.info('Rematch accepted', {
              requestId: payload.requestId,
              accepterId: socket.userId,
              newGameId: result.newGameId,
            });
          } else {
            // Decline rematch
            const result = await rematchService.declineRematch(payload.requestId, socket.userId);

            if (!result.success || !result.request) {
              this.emitError(
                socket,
                'INVALID_PAYLOAD',
                result.error || 'Failed to decline rematch',
                'rematch_respond'
              );
              return;
            }

            // Broadcast decline to all players
            this.io.to(result.request.gameId).emit('rematch_response', {
              requestId: payload.requestId,
              gameId: result.request.gameId,
              status: 'declined' as const,
            });

            logger.info('Rematch declined', {
              requestId: payload.requestId,
              declinerId: socket.userId,
            });
          }
        } catch (error) {
          if (error instanceof ZodError) {
            this.handleWebSocketValidationError(socket, 'rematch_respond', error);
            return;
          }

          logger.error('Error handling rematch_respond', {
            socketId: socket.id,
            userId: socket.userId,
            error: error instanceof Error ? error.message : String(error),
          });
          this.emitError(
            socket,
            'INTERNAL_ERROR',
            'Failed to process rematch response',
            'rematch_respond'
          );
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

      const isPlayer = socket.userId ? playerIds.includes(socket.userId) : false;
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
      this.gameRooms.get(gameId)?.add(socket.id);

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

      // Send chat history for this game
      try {
        const chatService = getChatPersistenceService();
        const chatMessages = await chatService.getMessagesForGame(gameId);
        if (chatMessages.length > 0) {
          socket.emit('chat_history', {
            gameId,
            messages: chatMessages.map((msg) => ({
              id: msg.id,
              gameId: msg.gameId,
              userId: msg.userId,
              username: msg.username,
              message: msg.message,
              createdAt: msg.createdAt.toISOString(),
            })),
          });
        }
      } catch (chatError) {
        // Non-fatal: log and continue without chat history
        logger.warn('Failed to load chat history on join', {
          gameId,
          userId: socket.userId,
          error: chatError instanceof Error ? chatError.message : String(chatError),
        });
      }

      // Notify others in the room - use appropriate event based on reconnection status
      if (isReconnection && pendingReconnection) {
        socket.to(gameId).emit('player_reconnected', {
          type: 'player_reconnected',
          data: {
            gameId,
            player: {
              id: socket.userId ?? '',
              username: socket.username ?? '',
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
              id: socket.userId ?? '',
              username: socket.username ?? '',
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
    const room = this.gameRooms.get(gameId);
    if (room) {
      room.delete(socket.id);
      if (room.size === 0) {
        this.gameRooms.delete(gameId);
      }
    }

    // Notify others in the room
    socket.to(gameId).emit('player_left', {
      type: 'player_left',
      data: {
        gameId,
        player: {
          id: socket.userId ?? '',
          username: socket.username ?? '',
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

  private async handlePlayerMove(socket: AuthenticatedSocket, data: PlayerMovePayload) {
    const { gameId, move } = data;

    if (!socket.gameId || socket.gameId !== gameId) {
      throw new Error('Not in game room');
    }

    // Wrap move processing in a lock to prevent race conditions
    await this.sessionManager.withGameLock(gameId, async () => {
      const session = await this.sessionManager.getOrCreateSession(gameId);
      // Type assertion needed: Zod-inferred types have subtle differences from
      // GameSession's PlayerMoveData due to exactOptionalPropertyTypes setting
      await session.handlePlayerMove(
        socket,
        move as Parameters<typeof session.handlePlayerMove>[1]
      );
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

  /**
   * HTTP harness entry point for applying a move via the same GameSession /
   * RulesBackendFacade pipeline as WebSocket moves. Used by
   * POST /api/games/:gameId/moves.
   */
  public async handlePlayerMoveFromHttp(
    gameId: string,
    userId: string,
    move: PlayerMovePayload['move']
  ): Promise<RulesResult> {
    return this.sessionManager.withGameLock(gameId, async () => {
      const session = await this.sessionManager.getOrCreateSession(gameId);
      // Type assertion needed: Zod-inferred types have subtle differences from
      // GameSession's PlayerMoveData due to exactOptionalPropertyTypes setting
      return session.handlePlayerMoveFromHttp(
        userId,
        move as Parameters<typeof session.handlePlayerMoveFromHttp>[1]
      );
    });
  }

  private async handleChatMessage(socket: AuthenticatedSocket, data: ChatMessagePayload) {
    const { gameId, text } = data;

    if (!gameId || !text) {
      throw new Error('Missing gameId or text');
    }

    if (!socket.userId) {
      throw new Error('User not authenticated');
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

    // Persist the chat message to the database
    try {
      const chatService = getChatPersistenceService();
      const savedMessage = await chatService.saveMessage({
        gameId,
        userId: socket.userId,
        message: text,
      });

      // Broadcast persisted chat message to all players in the game
      this.io.to(gameId).emit('chat_message_persisted', {
        id: savedMessage.id,
        gameId: savedMessage.gameId,
        userId: savedMessage.userId,
        username: savedMessage.username,
        message: savedMessage.message,
        createdAt: savedMessage.createdAt.toISOString(),
      });

      // Also emit legacy chat_message for backward compatibility
      this.io.to(gameId).emit('chat_message', {
        sender: savedMessage.username,
        text: savedMessage.message,
        timestamp: savedMessage.createdAt.toISOString(),
      });

      logger.info('Chat message persisted and sent', {
        messageId: savedMessage.id,
        userId: socket.userId,
        gameId,
        socketId: socket.id,
        messageLength: text.length,
      });
    } catch (error) {
      // If persistence fails, still broadcast the message (graceful degradation)
      logger.error('Failed to persist chat message, broadcasting anyway', {
        userId: socket.userId,
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });

      this.io.to(gameId).emit('chat_message', {
        sender: socket.username || 'Unknown',
        text,
        timestamp: new Date().toISOString(),
      });
    }
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

    // Remove from matchmaking queue if user was in queue
    if (socket.userId) {
      this.matchmakingService.removeFromQueue(socket.userId);
    }

    // Remove from user socket mapping immediately
    // (will be re-added if they reconnect)
    if (socket.userId) {
      this.userSockets.delete(socket.userId);
    }

    // Remove from game room tracking
    if (socket.gameId) {
      const disconnectRoom = this.gameRooms.get(socket.gameId);
      if (disconnectRoom) {
        disconnectRoom.delete(socket.id);
        if (disconnectRoom.size === 0) {
          this.gameRooms.delete(socket.gameId);
        }
      }

      // Notify others in the room
      socket.to(socket.gameId).emit('player_disconnected', {
        type: 'player_disconnected',
        data: {
          gameId: socket.gameId,
          player: {
            id: socket.userId ?? '',
            username: socket.username ?? '',
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
          const isSpectator = gameState.spectators.includes(userId);

          // Spectators do not participate in the reconnection timeout /
          // abandonment flow. When a spectator disconnects we simply clear
          // their connection diagnostics entry (if any) and avoid scheduling
          // a reconnection window.
          if (!player && isSpectator) {
            const key = `${gameId}:${userId}`;
            const hadState = this.playerConnectionStates.delete(key);

            if (hadState) {
              logger.info('Spectator disconnected without reconnection window', {
                userId,
                gameId,
              });
            }

            return;
          }

          // Only set up reconnection window for actual players, not spectators
          if (player && !isSpectator) {
            const reconnectionKey = `${gameId}:${userId}`;

            // Clear any existing timeout for this user/game combo
            const existing = this.pendingReconnections.get(reconnectionKey);
            if (existing) {
              clearTimeout(existing.timeout);
            }

            const timeout = setTimeout(() => {
              this.handleReconnectionTimeout(gameId, userId, player.playerNumber);
            }, config.server.wsReconnectionTimeoutMs);

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
              config.server.wsReconnectionTimeoutMs
            );
            this.playerConnectionStates.set(key, nextState);

            logger.info('Reconnection window started', {
              userId,
              gameId,
              playerNumber: player.playerNumber,
              timeoutMs: config.server.wsReconnectionTimeoutMs,
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

      // Best-effort game-level abandonment handling. This runs under the
      // per-game lock so that engine and persistence updates remain
      // consistent with any concurrent HTTP or WebSocket activity.
      void this.sessionManager.withGameLock(gameId, async () => {
        const lockedSession = this.sessionManager.getSession(gameId);
        if (!lockedSession) {
          return;
        }

        const state = lockedSession.getGameState();
        if (state.gameStatus !== 'active') {
          return;
        }

        const humanPlayers = state.players.filter((p) => p.type === 'human');
        if (humanPlayers.length === 0) {
          return;
        }

        const disconnectingPlayer = humanPlayers.find(
          (p) => p.playerNumber === playerNumber && p.id === userId
        );
        if (!disconnectingPlayer) {
          return;
        }

        const otherHumans = humanPlayers.filter((p) => p.id !== userId);

        // Determine whether at least one human opponent remains connected
        // or within their own reconnect window.
        const anyOpponentStillAlive = otherHumans.some((p) => {
          const s = this.playerConnectionStates.get(`${gameId}:${p.id}`);
          return s && s.kind !== 'disconnected_expired';
        });

        const shouldAwardWin = state.isRated && anyOpponentStillAlive;

        try {
          await lockedSession.handleAbandonmentForDisconnectedPlayer(playerNumber, shouldAwardWin);
        } catch (err) {
          logger.error('Failed to apply abandonment after reconnection timeout', {
            gameId,
            userId,
            playerNumber,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      });
    }
  }

  /**
   * Handle a clean resignation initiated via HTTP (e.g. POST /games/:gameId/leave)
   * by routing through the active GameSession so that GameEngine can produce a
   * canonical GameResult and GamePersistenceService.finishGame is invoked.
   */
  public async handlePlayerResignFromHttp(gameId: string, userId: string): Promise<void> {
    await this.sessionManager.withGameLock(gameId, async () => {
      const session = await this.sessionManager.getOrCreateSession(gameId);
      await session.handlePlayerResignationByUserId(userId);
    });
  }

  // Public methods for external use
  public sendToUser<E extends DefinedServerEvent>(
    userId: string,
    event: E,
    data: Parameters<ServerToClientEvents[E]>[0]
  ) {
    const socketId = this.userSockets.get(userId);
    if (socketId) {
      // Type assertion needed due to socket.io's complex DecorateAcknowledgements type
      (this.io.to(socketId) as { emit: (ev: E, d: typeof data) => void }).emit(event, data);
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
    // Best-effort cleanup of any in-memory GameSession instances associated
    // with this user so that session-scoped cancellation (AI turns, decision
    // timers) runs alongside socket teardown.
    const terminateSessionsForUser = (socket?: AuthenticatedSocket) => {
      const gameIds = new Set<string>();

      if (socket?.gameId) {
        gameIds.add(socket.gameId);
      }

      // Also scan connection state diagnostics for any games where this user
      // has a recorded connection snapshot.
      const suffix = `:${userId}`;
      for (const key of this.playerConnectionStates.keys()) {
        if (key.endsWith(suffix)) {
          const gameId = key.slice(0, -suffix.length);
          if (gameId) {
            gameIds.add(gameId);
          }
        }
      }

      for (const gameId of gameIds) {
        const session = this.sessionManager.getSession(gameId);
        // Check if session has terminate method (not part of public interface)
        const sessionWithTerminate = session as typeof session & {
          terminate?: (reason: string) => void;
        };
        if (sessionWithTerminate && typeof sessionWithTerminate.terminate === 'function') {
          try {
            sessionWithTerminate.terminate('session_cleanup');
          } catch (err) {
            logger.warn('Failed to terminate GameSession during WebSocket session termination', {
              userId,
              gameId,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }
      }
    };

    const socketId = this.userSockets.get(userId);
    if (!socketId) {
      // Even if there is no currently tracked socket, there may still be
      // in-memory sessions (for example after a prior disconnect). Attempt
      // session-level cleanup in a best-effort manner.
      terminateSessionsForUser();
      logger.info('No active WebSocket connections found for user', { userId, reason });
      return 0;
    }

    const socket = this.io.sockets.sockets.get(socketId) as AuthenticatedSocket | undefined;
    if (!socket) {
      // Socket ID was tracked but socket is no longer in the server's collection
      // Clean up the stale mapping
      this.userSockets.delete(userId);
      terminateSessionsForUser();
      logger.info('Stale socket mapping cleaned up for user', { userId, socketId, reason });
      return 0;
    }

    // Invoke GameSession termination before tearing down the transport so
    // that any session-scoped async work observes cancellation promptly.
    terminateSessionsForUser(socket);

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

  public sendToGame<E extends DefinedServerEvent>(
    gameId: string,
    event: E,
    data: Parameters<ServerToClientEvents[E]>[0]
  ) {
    // Type assertion needed due to socket.io's complex DecorateAcknowledgements type
    (this.io.to(gameId) as { emit: (ev: E, d: typeof data) => void }).emit(event, data);
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
    sessionStatus: unknown;
    lastAIRequestState: unknown;
    aiDiagnostics: unknown;
    connections: Record<string, PlayerConnectionState>;
    hasInMemorySession: boolean;
  } {
    const session = this.sessionManager.getSession(gameId);

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
  public broadcastLobbyEvent<E extends DefinedServerEvent>(
    event: E,
    data: Parameters<ServerToClientEvents[E]>[0]
  ) {
    // Type assertion needed due to socket.io's complex DecorateAcknowledgements type
    (this.io.to(WebSocketServer.LOBBY_ROOM) as { emit: (ev: E, d: typeof data) => void }).emit(
      event,
      data
    );
    logger.debug('Broadcast lobby event', { event, lobbyRoom: WebSocketServer.LOBBY_ROOM });
  }
}
