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
import {
  Move,
  Player,
  GameState,
  Position,
  AIProfile,
  BOARD_CONFIGS,
  TimeControl,
} from '../../shared/types/game';

export class GameSession {
  public readonly gameId: string;
  private io: SocketIOServer;
  private gameEngine!: GameEngine;
  private rulesFacade!: RulesBackendFacade;
  private interactionManager!: PlayerInteractionManager;
  private wsHandler!: WebSocketInteractionHandler;
  private pythonRulesClient: PythonRulesClient;
  private userSockets: Map<string, string>; // userId -> socketId

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

    const result = await this.rulesFacade.applyMove(engineMove);
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
      const state = this.gameEngine.getGameState();
      if (state.gameStatus !== 'active') return;

      const currentPlayerNumber = state.currentPlayer;
      const currentPlayer = state.players.find((p) => p.playerNumber === currentPlayerNumber);

      if (!currentPlayer || currentPlayer.type !== 'ai') return;

      let aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      if (!aiConfig) {
        const difficulty = currentPlayer.aiDifficulty ?? 5;
        globalAIEngine.createAI(currentPlayerNumber, difficulty);
        aiConfig = globalAIEngine.getAIConfig(currentPlayerNumber);
      }

      let result: { success: boolean; error?: string; gameState?: GameState; gameResult?: any };
      let appliedMoveType: Move['type'] | undefined;

      if (
        state.currentPhase === 'line_processing' ||
        state.currentPhase === 'territory_processing'
      ) {
        const allCandidates = this.gameEngine.getValidMoves(currentPlayerNumber);
        const decisionCandidates = allCandidates.filter((m) => {
          if (state.currentPhase === 'line_processing') {
            return m.type === 'process_line' || m.type === 'choose_line_reward';
          }
          return m.type === 'process_territory_region' || m.type === 'eliminate_rings_from_stack';
        });

        if (decisionCandidates.length === 0) return;

        const selected = globalAIEngine.chooseLocalMoveFromCandidates(
          currentPlayerNumber,
          state,
          decisionCandidates
        );

        if (!selected) return;

        const { id, timestamp, moveNumber, ...rest } = selected as any;
        const engineMove = rest as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
        appliedMoveType = engineMove.type;

        result = await this.rulesFacade.applyMove(engineMove);
      } else {
        const aiMove = await globalAIEngine.getAIMove(currentPlayerNumber, state);
        if (!aiMove) return;

        const { id, timestamp, moveNumber, ...rest } = aiMove;
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

      logger.info('AI move processed and applied', {
        gameId: this.gameId,
        playerNumber: currentPlayerNumber,
        moveType: appliedMoveType,
      });
    } catch (error) {
      logger.error('Error during AI turn', {
        gameId: this.gameId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
}
