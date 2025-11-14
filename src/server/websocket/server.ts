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
  PlayerChoiceResponse
} from '../../shared/types/game';
import { WebSocketInteractionHandler } from '../game/WebSocketInteractionHandler';
import { PlayerInteractionManager } from '../game/PlayerInteractionManager';

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
    this.io = new SocketIOServer(httpServer, {
      cors: {
        origin: process.env.CLIENT_URL || "http://localhost:3000",
        methods: ["GET", "POST"],
        credentials: true
      },
      transports: ['websocket', 'polling']
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
            isActive: true
          }
        });

        if (!user || !user.isActive) {
          return next(new Error('User not found or inactive'));
        }

        socket.userId = user.id;
        socket.username = user.username;
        
        logger.info('WebSocket authenticated', { 
          userId: user.id, 
          username: user.username,
          socketId: socket.id 
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
        socketId: socket.id 
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
          orderBy: { moveNumber: 'asc' }
        }
      }
    });

    if (!game) {
      throw new Error('Game not found');
    }

    // Create players array
    const players: Player[] = [];
    if (game.player1) {
      players.push({
        id: game.player1.id,
        username: game.player1.username,
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: game.timeControl ? JSON.parse(game.timeControl).initialTime : 600000,
        ringsInHand: BOARD_CONFIGS[game.boardType as keyof typeof BOARD_CONFIGS].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0
      });
    }
    if (game.player2) {
      players.push({
        id: game.player2.id,
        username: game.player2.username,
        playerNumber: 2,
        type: 'human',
        isReady: true,
        timeRemaining: game.timeControl ? JSON.parse(game.timeControl).initialTime : 600000,
        ringsInHand: BOARD_CONFIGS[game.boardType as keyof typeof BOARD_CONFIGS].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0
      });
    }

    // Create time control
    const timeControl: TimeControl = game.timeControl ?
      JSON.parse(game.timeControl) :
      { type: 'standard', initialTime: 600000, increment: 0 };

    // Map playerNumber -> Socket.IO target (typically a user socket id).
    const getTargetForPlayer = (playerNumber: number): string | undefined => {
      const player = players.find(p => p.playerNumber === playerNumber);
      if (!player) return undefined;
      return this.userSockets.get(player.id);
    };

    const wsHandler = new WebSocketInteractionHandler(
      this.io,
      gameId,
      getTargetForPlayer,
      30_000
    );
    const interactionManager = new PlayerInteractionManager(wsHandler);

    // Create game engine wired to the interaction manager for choices
    const gameEngine = new GameEngine(
      gameId,
      game.boardType as keyof typeof BOARD_CONFIGS,
      players,
      timeControl,
      (game as any).isRated ?? true,
      interactionManager
    );

    // Replay moves if any exist
    for (const move of game.moves) {
      const gameMove: Move = {
        id: move.id,
        type: move.moveType as any,
        player: parseInt(move.playerId),
        from: move.position ? JSON.parse(move.position).from : undefined,
        to: move.position ? JSON.parse(move.position).to : JSON.parse(move.position),
        timestamp: move.timestamp,
        thinkTime: 0,
        moveNumber: move.moveNumber
      };
      
      gameEngine.makeMove(gameMove);
    }

    this.gameEngines.set(gameId, gameEngine);
    this.interactionManagers.set(gameId, interactionManager);
    this.interactionHandlers.set(gameId, wsHandler);
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
        player4: { select: { id: true, username: true } }
      }
    });

    if (!game) {
      throw new Error('Game not found');
    }

    const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
      .filter(Boolean);
    
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
        validMoves: isPlayer ? gameEngine.getValidMoves(gameEngine.getGameState().currentPlayer) : []
      },
      timestamp: new Date().toISOString()
    });

    // Notify others in the room
    socket.to(gameId).emit('player_joined', {
      type: 'player_joined',
      data: {
        gameId,
        player: {
          id: socket.userId!,
          username: socket.username!
        }
      },
      timestamp: new Date().toISOString()
    });

    logger.info('Player joined game room', {
      userId: socket.userId,
      gameId,
      isPlayer,
      canSpectate,
      gamePhase: gameEngine.getGameState().currentPhase,
      currentPlayer: gameEngine.getGameState().currentPlayer
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
          username: socket.username!
        }
      },
      timestamp: new Date().toISOString()
    });

    delete socket.gameId;

    logger.info('Player left game room', { 
      userId: socket.userId, 
      gameId 
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

    // Verify it's the player's turn and move is valid
    const game = await prisma.game.findUnique({
      where: { id: gameId }
    });

    if (!game) {
      throw new Error('Game not found');
    }

    if (game.status !== 'ACTIVE') {
      throw new Error('Game is not active');
    }

    // TODO: Validate move with game engine
    // For now, just broadcast the move

    // Save move to database
    await prisma.move.create({
      data: {
        gameId,
        playerId: socket.userId!,
        moveNumber: move.moveNumber,
        position: move.position,
        moveType: move.moveType,
        timestamp: new Date()
      }
    });

    // Broadcast move to all players in the game
    this.io.to(gameId).emit('player_move', {
      type: 'player_move',
      data: {
        gameId,
        move: {
          ...move,
          playerId: socket.userId!,
          playerUsername: socket.username!
        }
      },
      timestamp: new Date().toISOString()
    });

    logger.info('Player move processed', { 
      userId: socket.userId, 
      gameId,
      move 
    });
  }

  private async handleChatMessage(socket: AuthenticatedSocket, data: any) {
    const { gameId, content } = data;

    if (!socket.gameId || socket.gameId !== gameId) {
      throw new Error('Not in game room');
    }

    // Broadcast chat message to all players in the game
    this.io.to(gameId).emit('chat_message', {
      type: 'chat_message',
      data: {
        gameId,
        message: {
          id: Date.now().toString(),
          playerId: socket.userId!,
          playerUsername: socket.username!,
          content: content,
          timestamp: new Date().toISOString()
        }
      },
      timestamp: new Date().toISOString()
    });

    logger.info('Chat message sent', { 
      userId: socket.userId, 
      gameId,
      messageLength: content.length 
    });
  }

  private handleDisconnect(socket: AuthenticatedSocket) {
    logger.info('WebSocket disconnected', { 
      userId: socket.userId, 
      username: socket.username,
      socketId: socket.id 
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
            username: socket.username!
          }
        },
        timestamp: new Date().toISOString()
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
