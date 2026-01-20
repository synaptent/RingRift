import {
  Prisma,
  Game as PrismaGame,
  Move as PrismaMove,
  MoveType as PrismaMoveType,
  GameStatus as PrismaGameStatus,
  BoardType as PrismaBoardType,
} from '@prisma/client';
import { z } from 'zod';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import {
  Move,
  MoveType,
  GameState,
  Position,
  BoardType,
  TimeControl,
  GameResult,
  GameStatus,
  RingStack,
  CaptureType,
  LineInfo,
  Territory,
} from '../../shared/types/game';
import { RatingService, RatingUpdateResult } from './RatingService';
import { getDisplayUsername, isDeletedUserUsername } from '../routes/user';

/**
 * Configuration for creating a new game in the database
 */
export interface CreateGameConfig {
  boardType: BoardType;
  maxPlayers: number;
  timeControl: TimeControl;
  isRated: boolean;
  allowSpectators?: boolean;
  rngSeed?: number;
  player1Id?: string;
  player2Id?: string;
  player3Id?: string;
  player4Id?: string;
  initialGameState?: Partial<GameState>;
}

/**
 * Player info returned from game queries, with display name handling for deleted users.
 */
export interface PlayerInfo {
  id: string;
  username: string;
  /** User-friendly display name ("Deleted Player" for anonymized users) */
  displayName: string;
  /** Whether this user has been deleted/anonymized */
  isDeleted: boolean;
}

/**
 * Result of loading a game from the database
 */
export interface LoadedGame {
  game: PrismaGame;
  moves: PrismaMove[];
  players: {
    player1?: PlayerInfo | null;
    player2?: PlayerInfo | null;
    player3?: PlayerInfo | null;
    player4?: PlayerInfo | null;
  };
}

/**
 * Summary of a game for listing purposes
 */
export interface GameSummary {
  id: string;
  boardType: BoardType;
  status: GameStatus;
  playerCount: number;
  maxPlayers: number;
  winnerId?: string | null;
  createdAt: Date;
  endedAt?: Date | null;
  moveCount: number;
}

/**
 * Move data to be persisted
 */
export interface SaveMoveData {
  gameId: string;
  playerId: string;
  moveNumber: number;
  move: Move;
}

/**
 * Service for persisting and loading game state to/from the database.
 *
 * This service handles:
 * - Creating new games with initial configuration
 * - Saving moves as they happen (async, non-blocking)
 * - Loading games from storage including move history
 * - Reconstructing game state from moves for replay
 * - Finishing games with final state and result
 *
 * Design decisions:
 * - Prisma client is obtained via getDatabaseClient() for flexibility
 * - All methods are static for ease of use without instance management
 * - Move saving is non-blocking to avoid impacting gameplay
 * - Rich move data is stored in the moveData JSON field
 */
export class GamePersistenceService {
  /**
   * Create a new game record in the database
   */
  static async createGame(config: CreateGameConfig): Promise<string> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      // Build data object with proper Prisma types
      // TimeControl is serialized to JSON for storage
      const data: Prisma.GameCreateInput = {
        boardType: config.boardType as PrismaBoardType,
        maxPlayers: config.maxPlayers,
        timeControl: JSON.parse(JSON.stringify(config.timeControl)) as Prisma.InputJsonValue,
        isRated: config.isRated,
        allowSpectators: config.allowSpectators ?? true,
        status: 'waiting' as PrismaGameStatus,
        gameState: config.initialGameState ? JSON.stringify(config.initialGameState) : '{}',
        ...(config.rngSeed !== undefined && { rngSeed: config.rngSeed }),
        ...(config.player1Id && { player1: { connect: { id: config.player1Id } } }),
        ...(config.player2Id && { player2: { connect: { id: config.player2Id } } }),
        ...(config.player3Id && { player3: { connect: { id: config.player3Id } } }),
        ...(config.player4Id && { player4: { connect: { id: config.player4Id } } }),
      };

      const game = await prisma.game.create({ data });

      logger.info('Game created in database', {
        gameId: game.id,
        boardType: config.boardType,
        maxPlayers: config.maxPlayers,
      });

      return game.id;
    } catch (error) {
      logger.error('Failed to create game in database', {
        error: error instanceof Error ? error.message : String(error),
        config,
      });
      throw error;
    }
  }

  /**
   * Save a move to the database (async, non-blocking)
   *
   * This method saves the move asynchronously to avoid blocking gameplay.
   * Errors are logged but not thrown to prevent gameplay disruption.
   */
  static async saveMove(data: SaveMoveData): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      logger.warn('Database not available for saving move', { gameId: data.gameId });
      return;
    }

    try {
      // Convert Move to position and moveData as JSON-serializable objects
      const position = JSON.parse(
        JSON.stringify({
          from: data.move.from,
          to: data.move.to,
        })
      );

      // Rich move data for replay and analysis
      const moveData = JSON.parse(JSON.stringify(this.serializeMoveData(data.move)));

      const createData: Prisma.MoveCreateInput = {
        game: { connect: { id: data.gameId } },
        player: { connect: { id: data.playerId } },
        moveNumber: data.moveNumber,
        position: position,
        moveType: this.mapMoveType(data.move.type),
        moveData: moveData,
        timestamp: data.move.timestamp,
      };

      await prisma.move.create({ data: createData });

      logger.debug('Move saved to database', {
        gameId: data.gameId,
        moveNumber: data.moveNumber,
        moveType: data.move.type,
      });
    } catch (error) {
      // Log but don't throw to avoid disrupting gameplay
      logger.error('Failed to save move to database', {
        gameId: data.gameId,
        moveNumber: data.moveNumber,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Save a move and return a promise that resolves when complete
   * Use this when you need to ensure the move is persisted
   */
  static async saveMoveSync(data: SaveMoveData): Promise<boolean> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return false;
    }

    try {
      // Convert Move to position and moveData as JSON-serializable objects
      const position = JSON.parse(
        JSON.stringify({
          from: data.move.from,
          to: data.move.to,
        })
      );

      const moveData = JSON.parse(JSON.stringify(this.serializeMoveData(data.move)));

      const createData: Prisma.MoveCreateInput = {
        game: { connect: { id: data.gameId } },
        player: { connect: { id: data.playerId } },
        moveNumber: data.moveNumber,
        position: position,
        moveType: this.mapMoveType(data.move.type),
        moveData: moveData,
        timestamp: data.move.timestamp,
      };

      await prisma.move.create({ data: createData });

      return true;
    } catch (error) {
      logger.error('Failed to save move synchronously', {
        gameId: data.gameId,
        moveNumber: data.moveNumber,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }

  /**
   * Format a player from database query to include display name handling.
   * @param player Raw player data from Prisma query
   * @returns PlayerInfo with display name and deleted status, or null
   */
  private static formatPlayer(player: { id: string; username: string } | null): PlayerInfo | null {
    if (!player) {
      return null;
    }
    const isDeleted = isDeletedUserUsername(player.username);
    return {
      id: player.id,
      username: player.username,
      displayName: getDisplayUsername(player.username),
      isDeleted,
    };
  }

  /**
   * Load a game from the database with all moves.
   * Player info includes display names that handle deleted/anonymized users.
   */
  static async loadGame(gameId: string): Promise<LoadedGame | null> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
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
        return null;
      }

      // Format players to include display names, handling deleted users gracefully
      return {
        game,
        moves: game.moves,
        players: {
          player1: this.formatPlayer(game.player1),
          player2: this.formatPlayer(game.player2),
          player3: this.formatPlayer(game.player3),
          player4: this.formatPlayer(game.player4),
        },
      };
    } catch (error) {
      logger.error('Failed to load game from database', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Get move history for a game
   */
  static async getGameHistory(gameId: string): Promise<Move[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const moves = await prisma.move.findMany({
        where: { gameId },
        orderBy: { moveNumber: 'asc' },
      });

      return moves.map((dbMove) => this.deserializeMove(dbMove));
    } catch (error) {
      logger.error('Failed to get game history', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Mark a game as completed with final state and result.
   * Also updates player ratings for rated games.
   *
   * @returns Rating update results if ratings were updated, undefined otherwise
   */
  static async finishGame(
    gameId: string,
    winnerId: string | null,
    finalState?: GameState,
    result?: GameResult
  ): Promise<{ ratingUpdates?: RatingUpdateResult[] }> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      // First fetch the game to get player IDs and rated status
      const existingGame = await prisma.game.findUnique({
        where: { id: gameId },
        select: {
          player1Id: true,
          player2Id: true,
          player3Id: true,
          player4Id: true,
          isRated: true,
          gameState: true,
        },
      });

      if (!existingGame) {
        throw new Error(`Game not found: ${gameId}`);
      }

      const updateData: Prisma.GameUpdateInput = {
        status: 'completed' as PrismaGameStatus,
        ...(winnerId && { winner: { connect: { id: winnerId } } }),
        endedAt: new Date(),
        updatedAt: new Date(),
      };

      if (finalState) {
        updateData.finalState = this.serializeGameState(finalState);
      }

      if (result) {
        // gameState is a Prisma Json field - it's already deserialized to an object
        const existingState = existingGame?.gameState;
        const parsedState =
          typeof existingState === 'string' ? JSON.parse(existingState) : existingState || {};
        updateData.gameState = JSON.stringify({ ...parsedState, result });
      }

      await prisma.game.update({
        where: { id: gameId },
        data: updateData,
      });

      logger.info('Game finished', {
        gameId,
        winnerId,
        hasResult: !!result,
        isRated: existingGame.isRated,
      });

      // Process rating updates for rated games. Abandonment-without-winner in
      // rated games is treated as a non-competitive termination and MUST NOT
      // change ratings (P18.3-1 §4.3), even though the game row itself may
      // still be marked isRated=true for auditing.
      let ratingUpdates: RatingUpdateResult[] | undefined;
      if (existingGame.isRated) {
        const reason = result?.reason;
        const skipForAbandonmentDraw = reason === 'abandonment' && !winnerId;

        if (!skipForAbandonmentDraw) {
          const playerIds = [
            existingGame.player1Id,
            existingGame.player2Id,
            existingGame.player3Id,
            existingGame.player4Id,
          ].filter((id): id is string => !!id);

          if (playerIds.length >= 2) {
            try {
              ratingUpdates = await RatingService.processGameResult(gameId, winnerId, playerIds);
              logger.info('Ratings updated after game completion', {
                gameId,
                updates: ratingUpdates,
              });
            } catch (ratingError) {
              // Log but don't fail the game completion if rating update fails
              logger.error('Failed to update ratings after game completion', {
                gameId,
                error: ratingError instanceof Error ? ratingError.message : String(ratingError),
              });
            }
          }
        } else {
          logger.info('Skipping rating update for abandonment without winner', {
            gameId,
            reason,
          });
        }
      }

      // Only include ratingUpdates in result if defined
      if (ratingUpdates) {
        return { ratingUpdates };
      }
      return {};
    } catch (error) {
      logger.error('Failed to finish game', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Update game status (e.g., waiting -> active)
   */
  static async updateGameStatus(gameId: string, status: GameStatus): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const updateData: Prisma.GameUpdateInput = {
        status: status as PrismaGameStatus,
        updatedAt: new Date(),
        ...(status === 'active' && { startedAt: new Date() }),
      };

      await prisma.game.update({
        where: { id: gameId },
        data: updateData,
      });

      logger.debug('Game status updated', { gameId, status });
    } catch (error) {
      logger.error('Failed to update game status', {
        gameId,
        status,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Update the current game state snapshot
   */
  static async updateGameState(gameId: string, gameState: GameState): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return;
    }

    try {
      // First, read the existing game state to preserve metadata fields
      // (aiOpponents, rulesOptions) that were set at game creation time
      const existingGame = await prisma.game.findUnique({
        where: { id: gameId },
        select: { gameState: true },
      });

      // Extract preserved metadata from existing state
      const existingState =
        typeof existingGame?.gameState === 'string'
          ? JSON.parse(existingGame.gameState)
          : existingGame?.gameState;
      const aiOpponents = existingState?.aiOpponents;
      const rulesOptions = existingState?.rulesOptions;

      // Serialize game state while preserving metadata
      const serialized = this.serializeGameState(gameState);
      const serializedObj = JSON.parse(serialized);

      // Re-add preserved metadata
      if (aiOpponents) serializedObj.aiOpponents = aiOpponents;
      if (rulesOptions) serializedObj.rulesOptions = rulesOptions;

      await prisma.game.update({
        where: { id: gameId },
        data: {
          gameState: JSON.stringify(serializedObj),
          updatedAt: new Date(),
        },
      });
    } catch (error) {
      logger.error('Failed to update game state', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Update game state snapshot including internal engine state for crash recovery.
   * This is called after each move to persist state that cannot be reconstructed
   * from move history alone (e.g., decision phase internal flags).
   *
   * This is an async, non-blocking operation. Failures are logged but do not
   * affect gameplay.
   */
  static async updateGameStateWithInternal(
    gameId: string,
    gameState: GameState,
    internalState: {
      hasPlacedThisTurn?: boolean | undefined;
      mustMoveFromStackKey?: string | undefined;
      chainCaptureState?: unknown;
      pendingTerritorySelfElimination?: boolean | undefined;
      pendingLineRewardElimination?: boolean | undefined;
      swapSidesApplied?: boolean | undefined;
    }
  ): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return;
    }

    try {
      // First, read the existing game state to preserve metadata fields
      // (aiOpponents, rulesOptions) that were set at game creation time
      const existingGame = await prisma.game.findUnique({
        where: { id: gameId },
        select: { gameState: true },
      });

      // Extract preserved metadata from existing state
      const existingState =
        typeof existingGame?.gameState === 'string'
          ? JSON.parse(existingGame.gameState)
          : existingGame?.gameState;
      const aiOpponents = existingState?.aiOpponents;
      const rulesOptions = existingState?.rulesOptions;

      // Serialize with internal state included, preserving metadata
      const serializable = {
        ...gameState,
        board: {
          ...gameState.board,
          stacks: Array.from(gameState.board.stacks.entries()),
          markers: Array.from(gameState.board.markers.entries()),
          collapsedSpaces: Array.from(gameState.board.collapsedSpaces.entries()),
          territories: Array.from(gameState.board.territories.entries()),
        },
        // Include internal state for crash recovery
        _internalState: internalState,
        // Preserve metadata from game creation
        ...(aiOpponents && { aiOpponents }),
        ...(rulesOptions && { rulesOptions }),
      };

      await prisma.game.update({
        where: { id: gameId },
        data: {
          gameState: JSON.stringify(serializable),
          updatedAt: new Date(),
        },
      });
    } catch (error) {
      logger.error('Failed to update game state with internal state', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Get recent games for a user
   */
  static async getUserGames(userId: string, limit = 10): Promise<GameSummary[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const games = await prisma.game.findMany({
        where: {
          OR: [
            { player1Id: userId },
            { player2Id: userId },
            { player3Id: userId },
            { player4Id: userId },
          ],
        },
        orderBy: { createdAt: 'desc' },
        take: limit,
        include: {
          _count: {
            select: { moves: true },
          },
        },
      });

      return games.map((game) => ({
        id: game.id,
        boardType: game.boardType as BoardType,
        status: game.status as GameStatus,
        playerCount: [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
          Boolean
        ).length,
        maxPlayers: game.maxPlayers,
        winnerId: game.winnerId,
        createdAt: game.createdAt,
        endedAt: game.endedAt,
        moveCount: game._count.moves,
      }));
    } catch (error) {
      logger.error('Failed to get user games', {
        userId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Get active games (for server restart recovery)
   */
  static async getActiveGames(): Promise<string[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const games = await prisma.game.findMany({
        where: {
          status: 'active' as PrismaGameStatus,
        },
        select: { id: true },
      });

      return games.map((g) => g.id);
    } catch (error) {
      logger.error('Failed to get active games', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Check if a game exists
   */
  static async gameExists(gameId: string): Promise<boolean> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return false;
    }

    try {
      const game = await prisma.game.findUnique({
        where: { id: gameId },
        select: { id: true },
      });
      return !!game;
    } catch (_error) {
      return false;
    }
  }

  /**
   * Delete a game and all its moves (for cleanup/testing)
   */
  static async deleteGame(gameId: string): Promise<void> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      // Moves will be deleted via cascade
      await prisma.game.delete({
        where: { id: gameId },
      });

      logger.info('Game deleted', { gameId });
    } catch (error) {
      logger.error('Failed to delete game', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  // =====================
  // Private Helper Methods
  // =====================

  /**
   * Map domain MoveType to Prisma MoveType
   */
  private static mapMoveType(type: MoveType): PrismaMoveType {
    // Cast directly since we've aligned the Prisma enum with domain types
    // The Prisma enum now contains all the same values as the domain MoveType
    return type as unknown as PrismaMoveType;
  }

  /**
   * Serialize a Move to rich moveData JSON
   */
  private static serializeMoveData(move: Move): Record<string, unknown> {
    // Include all relevant move properties for replay
    const base: Record<string, unknown> = {
      id: move.id,
      type: move.type,
      player: move.player,
      from: move.from,
      to: move.to,
      thinkTime: move.thinkTime,
      moveNumber: move.moveNumber,
      // Optional fields
      ...(move.buildAmount !== undefined && { buildAmount: move.buildAmount }),
      ...(move.placedOnStack !== undefined && { placedOnStack: move.placedOnStack }),
      ...(move.placementCount !== undefined && { placementCount: move.placementCount }),
      ...(move.stackMoved && { stackMoved: move.stackMoved }),
      ...(move.minimumDistance !== undefined && { minimumDistance: move.minimumDistance }),
      ...(move.actualDistance !== undefined && { actualDistance: move.actualDistance }),
      ...(move.markerLeft && { markerLeft: move.markerLeft }),
      ...(move.captureType && { captureType: move.captureType }),
      ...(move.captureTarget && { captureTarget: move.captureTarget }),
      ...(move.capturedStacks && { capturedStacks: move.capturedStacks }),
      ...(move.captureChain && { captureChain: move.captureChain }),
      ...(move.overtakenRings && { overtakenRings: move.overtakenRings }),
      ...(move.formedLines && { formedLines: move.formedLines }),
      ...(move.collapsedMarkers && { collapsedMarkers: move.collapsedMarkers }),
      ...(move.claimedTerritory && { claimedTerritory: move.claimedTerritory }),
      ...(move.disconnectedRegions && { disconnectedRegions: move.disconnectedRegions }),
      ...(move.eliminatedRings && { eliminatedRings: move.eliminatedRings }),
      ...(move.eliminationFromStack && { eliminationFromStack: move.eliminationFromStack }),
    };

    // Persist any attached decision auto-resolve metadata when present.
    // This is used by the /games/:gameId/history HTTP route to surface
    // autoResolved badges in the history UI without requiring a schema
    // migration (the data lives entirely inside moveData JSON).
    const moveWithMeta = move as Move & { decisionAutoResolved?: unknown };
    if (moveWithMeta.decisionAutoResolved) {
      base.decisionAutoResolved = moveWithMeta.decisionAutoResolved;
    }

    return base;
  }

  /**
   * Deserialize a database move record to domain Move
   */
  private static deserializeMove(dbMove: PrismaMove): Move {
    const moveData = (dbMove as { moveData?: unknown }).moveData as Record<string, unknown> | null;
    const position = dbMove.position as { from?: Position; to?: Position } | null;

    // If we have rich moveData, use it
    if (moveData && typeof moveData === 'object') {
      const result: Move = {
        id: (moveData.id as string) || dbMove.id,
        type: (moveData.type as MoveType) || (dbMove.moveType as MoveType),
        player: moveData.player as number,
        to: (moveData.to as Position) || position?.to || { x: 0, y: 0 },
        timestamp: dbMove.timestamp,
        thinkTime: (moveData.thinkTime as number) || 0,
        moveNumber: dbMove.moveNumber,
      };

      // Add optional fields only if they exist
      if (moveData.from) result.from = moveData.from as Position;
      if (moveData.buildAmount !== undefined) result.buildAmount = moveData.buildAmount as number;
      if (moveData.placedOnStack !== undefined)
        result.placedOnStack = moveData.placedOnStack as boolean;
      if (moveData.placementCount !== undefined)
        result.placementCount = moveData.placementCount as number;
      if (moveData.stackMoved) result.stackMoved = moveData.stackMoved as RingStack;
      if (moveData.minimumDistance !== undefined)
        result.minimumDistance = moveData.minimumDistance as number;
      if (moveData.actualDistance !== undefined)
        result.actualDistance = moveData.actualDistance as number;
      if (moveData.markerLeft) result.markerLeft = moveData.markerLeft as Position;
      if (moveData.captureType) result.captureType = moveData.captureType as CaptureType;
      if (moveData.captureTarget) result.captureTarget = moveData.captureTarget as Position;
      if (moveData.capturedStacks) result.capturedStacks = moveData.capturedStacks as RingStack[];
      if (moveData.captureChain) result.captureChain = moveData.captureChain as Position[];
      if (moveData.overtakenRings) result.overtakenRings = moveData.overtakenRings as number[];
      if (moveData.formedLines) result.formedLines = moveData.formedLines as LineInfo[];
      if (moveData.collapsedMarkers)
        result.collapsedMarkers = moveData.collapsedMarkers as Position[];
      if (moveData.claimedTerritory)
        result.claimedTerritory = moveData.claimedTerritory as Territory[];
      if (moveData.disconnectedRegions)
        result.disconnectedRegions = moveData.disconnectedRegions as Territory[];
      if (moveData.eliminatedRings)
        result.eliminatedRings = moveData.eliminatedRings as { player: number; count: number }[];
      if (moveData.eliminationFromStack)
        result.eliminationFromStack = moveData.eliminationFromStack as {
          position: Position;
          capHeight: number;
          totalHeight: number;
        };

      return result;
    }

    // Fallback: reconstruct from minimal data
    const fallbackResult: Move = {
      id: dbMove.id,
      type: dbMove.moveType as MoveType,
      player: 0, // Will need to be inferred from playerId
      to: position?.to || { x: 0, y: 0 },
      timestamp: dbMove.timestamp,
      thinkTime: 0,
      moveNumber: dbMove.moveNumber,
    };

    if (position?.from) {
      fallbackResult.from = position.from;
    }

    return fallbackResult;
  }

  /**
   * Serialize GameState for storage
   */
  private static serializeGameState(state: GameState): string {
    // Convert Maps to serializable format
    const serializable = {
      ...state,
      board: {
        ...state.board,
        stacks: Array.from(state.board.stacks.entries()),
        markers: Array.from(state.board.markers.entries()),
        collapsedSpaces: Array.from(state.board.collapsedSpaces.entries()),
        territories: Array.from(state.board.territories.entries()),
      },
    };
    return JSON.stringify(serializable);
  }

  /**
   * Zod schema for validating deserialized game state structure.
   * Uses passthrough() to allow additional fields for forward compatibility.
   */
  private static readonly GameStateSchema = z
    .object({
      id: z.string(),
      boardType: z.string(),
      numPlayers: z.number(),
      currentPlayer: z.number(),
      currentPhase: z.string(),
      turnNumber: z.number(),
      gameStatus: z.string(),
      board: z
        .object({
          type: z.string(),
          // stacks, markers, collapsedSpaces, territories are arrays (serialized Maps)
          stacks: z.array(z.tuple([z.string(), z.any()])).optional(),
          markers: z.array(z.tuple([z.string(), z.any()])).optional(),
          collapsedSpaces: z.array(z.tuple([z.string(), z.any()])).optional(),
          territories: z.array(z.tuple([z.string(), z.any()])).optional(),
        })
        .passthrough(),
      players: z.array(z.any()),
    })
    .passthrough();

  /**
   * Deserialize GameState from storage with validation.
   * @throws Error if JSON is invalid or doesn't match expected schema
   */
  static deserializeGameState(json: string): GameState {
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch (parseError) {
      const message = parseError instanceof Error ? parseError.message : String(parseError);
      logger.error('Failed to parse game state JSON', { error: message, jsonLength: json.length });
      throw new Error(`Invalid game state JSON: ${message}`);
    }

    // Validate basic structure
    const validationResult = this.GameStateSchema.safeParse(parsed);
    if (!validationResult.success) {
      const issues = validationResult.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`);
      logger.error('Game state validation failed', {
        errors: issues,
        parsedKeys: parsed && typeof parsed === 'object' ? Object.keys(parsed) : [],
      });
      throw new Error(`Game state validation failed: ${issues.join(', ')}`);
    }

    const validated = validationResult.data;

    // Reconstruct Maps from serialized arrays.
    // We create a new board object since Zod's types are strict about array vs Map.
    const reconstructedBoard = {
      ...validated.board,
      stacks: Array.isArray(validated.board.stacks)
        ? new Map(validated.board.stacks)
        : validated.board.stacks,
      markers: Array.isArray(validated.board.markers)
        ? new Map(validated.board.markers)
        : validated.board.markers,
      collapsedSpaces: Array.isArray(validated.board.collapsedSpaces)
        ? new Map(validated.board.collapsedSpaces)
        : validated.board.collapsedSpaces,
      territories: Array.isArray(validated.board.territories)
        ? new Map(validated.board.territories)
        : validated.board.territories,
    };

    // Cast through unknown since passthrough() preserves additional GameState fields
    // that aren't in our validation schema
    return {
      ...validated,
      board: reconstructedBoard,
    } as unknown as GameState;
  }
}

export default GamePersistenceService;
