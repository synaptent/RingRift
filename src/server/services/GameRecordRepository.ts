/**
 * GameRecordRepository - CRUD operations for game records
 *
 * Provides database access for storing and retrieving completed game records,
 * supporting:
 * - Online game completion storage
 * - Self-play game recording (CMA-ES, soak tests)
 * - Training data export
 * - Replay system data access
 */

import { PrismaClient, Game, BoardType, Prisma } from '@prisma/client';
import type { JsonValue } from '@prisma/client/runtime/client';
import { getDatabaseClient } from '../database/connection';
import {
  GameRecord,
  MoveRecord,
  GameRecordMetadata,
  FinalScore,
  GameOutcome,
  RecordSource,
  PlayerRecordInfo,
  gameRecordToJsonlLine,
} from '../../shared/types/gameRecord';
import { GameState, MoveType, Position, LineInfo, Territory } from '../../shared/types/game';
import { logger } from '../utils/logger';

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface GameRecordFilter {
  boardType?: BoardType;
  numPlayers?: number;
  outcome?: GameOutcome;
  source?: RecordSource;
  isRated?: boolean;
  tags?: string[];
  fromDate?: Date;
  toDate?: Date;
  playerId?: string;
  limit?: number;
  offset?: number;
}

export interface GameRecordSummary {
  id: string;
  boardType: BoardType;
  numPlayers: number;
  winner: number | null;
  outcome: GameOutcome;
  totalMoves: number;
  totalDurationMs: number;
  startedAt: Date;
  endedAt: Date;
  source: RecordSource;
}

// Type for Prisma query result with includes - uses Prisma's actual types
type GameWithRelations = Prisma.GameGetPayload<{
  include: {
    moves: {
      include: { player: { select: { username: true } } };
    };
    player1: { select: { username: true; rating: true } };
    player2: { select: { username: true; rating: true } };
    player3: { select: { username: true; rating: true } };
    player4: { select: { username: true; rating: true } };
    winner: { select: { username: true } };
  };
}>;

/**
 * Type guard to check if a game has complete record data
 */
function isCompletedGameRecord(
  game: Prisma.GameGetPayload<object> | null
): game is Prisma.GameGetPayload<object> & { finalState: JsonValue; outcome: string } {
  return game !== null && game.finalState !== null && game.outcome !== null;
}

// ────────────────────────────────────────────────────────────────────────────
// Repository
// ────────────────────────────────────────────────────────────────────────────

export class GameRecordRepository {
  private getDb(): PrismaClient {
    const db = getDatabaseClient();
    if (!db) {
      throw new Error('Database not connected');
    }
    return db;
  }

  /**
   * Save a completed game as a GameRecord.
   *
   * This updates the Game row with:
   * - finalState: Complete GameState snapshot
   * - finalScore: Per-player score breakdown
   * - outcome: How the game ended
   * - recordMetadata: Training pipeline metadata
   */
  async saveGameRecord(
    gameId: string,
    finalState: GameState,
    outcome: GameOutcome,
    finalScore: FinalScore,
    metadata: Partial<GameRecordMetadata> = {}
  ): Promise<void> {
    const db = this.getDb();

    const recordMetadata: GameRecordMetadata = {
      recordVersion: '1.0.0',
      createdAt: new Date(),
      source: metadata.source ?? 'online_game',
      tags: metadata.tags ?? [],
      ...(metadata.sourceId !== undefined && { sourceId: metadata.sourceId }),
      ...(metadata.generation !== undefined && { generation: metadata.generation }),
      ...(metadata.candidateId !== undefined && { candidateId: metadata.candidateId }),
      // Human vs AI training metadata (January 2026)
      ...(metadata.humanPlayer !== undefined && { humanPlayer: metadata.humanPlayer }),
      ...(metadata.aiDifficulty !== undefined && { aiDifficulty: metadata.aiDifficulty }),
      ...(metadata.aiType !== undefined && { aiType: metadata.aiType }),
      ...(metadata.humanWon !== undefined && { humanWon: metadata.humanWon }),
      ...(metadata.eligibleForTraining !== undefined && {
        eligibleForTraining: metadata.eligibleForTraining,
      }),
    };

    // Use type assertion to work around Prisma client not yet having
    // the new record fields in its generated types. The actual database
    // schema has these fields after migration.
    await db.game.update({
      where: { id: gameId },
      data: {
        finalState: JSON.parse(JSON.stringify(finalState)),
        finalScore: JSON.parse(JSON.stringify(finalScore)),
        outcome,
        recordMetadata: JSON.parse(JSON.stringify(recordMetadata)),
        endedAt: new Date(),
      } as Parameters<typeof db.game.update>[0]['data'],
    });

    logger.info('Game record saved', { gameId, outcome, source: recordMetadata.source });
  }

  /**
   * Load a complete GameRecord by ID.
   *
   * Returns null if the game doesn't exist or hasn't been completed.
   */
  async getGameRecord(gameId: string): Promise<GameRecord | null> {
    const db = this.getDb();

    const game = await db.game.findUnique({
      where: { id: gameId },
      include: {
        moves: {
          orderBy: { moveNumber: 'asc' },
          include: { player: { select: { username: true } } },
        },
        player1: { select: { username: true, rating: true } },
        player2: { select: { username: true, rating: true } },
        player3: { select: { username: true, rating: true } },
        player4: { select: { username: true, rating: true } },
        winner: { select: { username: true } },
      },
    });

    // Use type guard to verify the game has complete record data
    if (!isCompletedGameRecord(game)) {
      return null;
    }

    // At this point, TypeScript knows game has the required fields
    return this.gameToGameRecord(game as GameWithRelations);
  }

  /**
   * List game records with optional filtering.
   */
  async listGameRecords(filter: GameRecordFilter = {}): Promise<GameRecordSummary[]> {
    const db = this.getDb();

    const where = this.buildCompletedGamesWhereClause(filter);

    const games = await db.game.findMany({
      where,
      include: {
        _count: { select: { moves: true } },
      },
      orderBy: { endedAt: 'desc' },
      take: filter.limit ?? 50,
      skip: filter.offset ?? 0,
    });

    return games.map((game) => this.gameToSummary(game));
  }

  /**
   * Export game records as JSONL for training pipelines.
   *
   * Returns an async generator that yields one JSONL line per game.
   */
  async *exportAsJsonl(filter: GameRecordFilter = {}): AsyncGenerator<string> {
    const db = this.getDb();
    const batchSize = 100;
    const where = this.buildCompletedGamesWhereClause(filter);
    let offset = filter.offset ?? 0;
    const hasLimit = typeof filter.limit === 'number';
    let remaining = hasLimit ? (filter.limit as number) : 0;

    while (true) {
      if (hasLimit && remaining <= 0) {
        break;
      }

      const take = hasLimit ? Math.min(batchSize, remaining) : batchSize;

      const games = await db.game.findMany({
        where,
        include: {
          moves: {
            orderBy: { moveNumber: 'asc' },
            include: { player: { select: { username: true } } },
          },
          player1: { select: { username: true, rating: true } },
          player2: { select: { username: true, rating: true } },
          player3: { select: { username: true, rating: true } },
          player4: { select: { username: true, rating: true } },
          winner: { select: { username: true } },
        },
        orderBy: { endedAt: 'desc' },
        take,
        skip: offset,
      });

      if (games.length === 0) break;

      for (const game of games) {
        // Use type guard to verify complete record data
        if (isCompletedGameRecord(game)) {
          const record = this.gameToGameRecord(game as GameWithRelations);
          yield gameRecordToJsonlLine(record);

          if (hasLimit) {
            remaining -= 1;
            if (remaining <= 0) {
              return;
            }
          }
        }
      }

      offset += games.length;
    }
  }

  /**
   * Count total game records matching filter.
   */
  async countGameRecords(filter: GameRecordFilter = {}): Promise<number> {
    const db = this.getDb();

    const where = this.buildCompletedGamesWhereClause(filter);

    return db.game.count({ where });
  }

  /**
   * Delete old game records for data retention.
   */
  async deleteOldRecords(beforeDate: Date, source?: RecordSource): Promise<number> {
    const db = this.getDb();

    const where: Prisma.GameWhereInput = {
      endedAt: { lt: beforeDate },
      finalState: { not: Prisma.DbNull },
    };

    // Note: Filtering by source in JSON requires raw query or jsonPath
    // For now, delete all records before the date (source param logged for future use)

    const result = await db.game.deleteMany({ where });
    logger.info('Deleted old game records', {
      deletedCount: result.count,
      beforeDate: beforeDate.toISOString(),
      source,
    });
    return result.count;
  }

  // ────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ────────────────────────────────────────────────────────────────────────

  private buildCompletedGamesWhereClause(filter: GameRecordFilter = {}): Prisma.GameWhereInput {
    const where: Prisma.GameWhereInput = {
      finalState: { not: Prisma.DbNull },
      outcome: { not: null },
    };

    if (filter.boardType) where.boardType = filter.boardType;
    if (filter.numPlayers) where.maxPlayers = filter.numPlayers;
    if (filter.outcome) where.outcome = filter.outcome;
    if (filter.isRated !== undefined) where.isRated = filter.isRated;

    if (filter.fromDate || filter.toDate) {
      const dateFilter: Prisma.DateTimeFilter<'Game'> = {};
      if (filter.fromDate) dateFilter.gte = filter.fromDate;
      if (filter.toDate) dateFilter.lte = filter.toDate;
      where.endedAt = dateFilter;
    }

    if (filter.playerId) {
      where.OR = [
        { player1Id: filter.playerId },
        { player2Id: filter.playerId },
        { player3Id: filter.playerId },
        { player4Id: filter.playerId },
      ];
    }

    // Note: Filtering by source/tags in JSON recordMetadata would require
    // JSON-path queries; export helpers currently ignore these filters.
    return where;
  }

  private gameToGameRecord(game: GameWithRelations): GameRecord {
    const metadata = (game.recordMetadata ?? {
      recordVersion: '1.0.0',
      createdAt: game.endedAt ?? new Date(),
      source: 'online_game' as RecordSource,
      tags: [],
    }) as GameRecordMetadata;

    const players: PlayerRecordInfo[] = [];
    const playerRefs = [game.player1, game.player2, game.player3, game.player4];

    // Extract AI opponents config from gameState for populating AI player metadata
    const gameStateRaw = game.gameState as Record<string, unknown> | null;
    const aiOpponents = gameStateRaw?.aiOpponents as
      | { count: number; difficulty: number[]; mode?: string; aiType?: string; aiTypes?: string[] }
      | undefined;

    // Track AI slot index for mapping to difficulty array
    let aiSlotIndex = 0;

    for (let i = 0; i < game.maxPlayers; i++) {
      const user = playerRefs[i];
      if (user) {
        // Human player
        players.push({
          playerNumber: i + 1,
          username: user.username,
          playerType: 'human',
          ratingBefore: user.rating,
        });
      } else {
        // AI player - extract difficulty and type from aiOpponents config
        const aiDifficulty = aiOpponents?.difficulty?.[aiSlotIndex];
        // Use per-player AI type from aiTypes array if available, falling back to shared aiType
        const aiType = aiOpponents?.aiTypes?.[aiSlotIndex] ?? aiOpponents?.aiType;
        players.push({
          playerNumber: i + 1,
          username: `AI Player ${i + 1}`,
          playerType: 'ai',
          ...(aiDifficulty !== undefined && { aiDifficulty }),
          ...(aiType !== undefined && { aiType }),
        });
        aiSlotIndex++;
      }
    }

    const moves: MoveRecord[] = game.moves.map((move) => {
      const raw = (move.moveData ?? {}) as Record<string, unknown>;

      const seatFromMove = typeof raw.player === 'number' ? (raw.player as number) : undefined;
      const playerNumber =
        seatFromMove && seatFromMove >= 1 && seatFromMove <= game.maxPlayers
          ? seatFromMove
          : players.findIndex((p) => p.username === move.player.username) + 1 || 1;

      const thinkTimeMs = typeof raw.thinkTime === 'number' ? (raw.thinkTime as number) : 0;

      const from = raw.from as Position | undefined;
      const to = raw.to as Position | undefined;
      const captureTarget = raw.captureTarget as Position | undefined;

      const placementCount =
        typeof raw.placementCount === 'number' ? (raw.placementCount as number) : undefined;
      const placedOnStack =
        typeof raw.placedOnStack === 'boolean' ? (raw.placedOnStack as boolean) : undefined;

      const formedLines =
        Array.isArray(raw.formedLines) && raw.formedLines.length > 0
          ? (raw.formedLines as LineInfo[])
          : undefined;
      const collapsedMarkers =
        Array.isArray(raw.collapsedMarkers) && raw.collapsedMarkers.length > 0
          ? (raw.collapsedMarkers as Position[])
          : undefined;
      const disconnectedRegions =
        Array.isArray(raw.disconnectedRegions) && raw.disconnectedRegions.length > 0
          ? (raw.disconnectedRegions as Territory[])
          : undefined;
      const eliminatedRings =
        Array.isArray(raw.eliminatedRings) && raw.eliminatedRings.length > 0
          ? (raw.eliminatedRings as { player: number; count: number }[])
          : undefined;

      const rrn =
        typeof raw.rrn === 'string' && (raw.rrn as string).length > 0
          ? (raw.rrn as string)
          : undefined;

      return {
        moveNumber: move.moveNumber,
        player: playerNumber,
        type: move.moveType as MoveType,
        thinkTimeMs,
        ...(from !== undefined && { from }),
        ...(to !== undefined && { to }),
        ...(captureTarget !== undefined && { captureTarget }),
        ...(placementCount !== undefined && { placementCount }),
        ...(placedOnStack !== undefined && { placedOnStack }),
        ...(formedLines !== undefined && { formedLines }),
        ...(collapsedMarkers !== undefined && { collapsedMarkers }),
        ...(disconnectedRegions !== undefined && { disconnectedRegions }),
        ...(eliminatedRings !== undefined && { eliminatedRings }),
        ...(rrn !== undefined && { rrn }),
      };
    });

    const winnerIndex = game.winner
      ? players.findIndex((p) => p.username === game.winner?.username) + 1
      : undefined;

    const startedAt = game.startedAt ?? game.createdAt;
    const endedAt = game.endedAt ?? new Date();
    const durationMs = endedAt.getTime() - startedAt.getTime();

    const record: GameRecord = {
      id: game.id,
      boardType: game.boardType,
      numPlayers: game.maxPlayers,
      isRated: game.isRated,
      players,
      outcome: game.outcome as GameOutcome,
      finalScore: (game.finalScore ?? {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      }) as FinalScore,
      startedAt,
      endedAt,
      totalMoves: moves.length,
      totalDurationMs: durationMs,
      moves,
      metadata,
    };

    // Only add optional fields if they have values
    if (game.rngSeed !== null) {
      record.rngSeed = game.rngSeed;
    }
    if (winnerIndex !== undefined) {
      record.winner = winnerIndex;
    }

    return record;
  }

  private gameToSummary(game: Game & { _count: { moves: number } }): GameRecordSummary {
    const metadata = (game.recordMetadata ?? {}) as Partial<GameRecordMetadata>;
    const startedAt = game.startedAt ?? game.createdAt;
    const endedAt = game.endedAt ?? new Date();

    return {
      id: game.id,
      boardType: game.boardType,
      numPlayers: game.maxPlayers,
      winner: null, // Would need to resolve from winnerId
      outcome: (game.outcome ?? 'abandonment') as GameOutcome,
      totalMoves: game._count.moves,
      totalDurationMs: endedAt.getTime() - startedAt.getTime(),
      startedAt,
      endedAt,
      source: metadata.source ?? 'online_game',
    };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Human vs AI Game Export (January 2026)
  // Used by training cluster to import valuable human game data
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * Build a where clause for human vs AI games.
   * Human vs AI = at least one player slot is null (AI) and at least one is not null (human)
   */
  private buildHumanVsAiWhereClause(filter: GameRecordFilter = {}): Prisma.GameWhereInput {
    const where: Prisma.GameWhereInput = {
      // Game must be completed
      status: 'completed',
      outcome: { not: null },
      finalState: { not: Prisma.JsonNull },
      // At least one human player (non-null player slot)
      OR: [
        { player1Id: { not: null } },
        { player2Id: { not: null } },
        { player3Id: { not: null } },
        { player4Id: { not: null } },
      ],
      // At least one AI player (null player slot for an active position)
      // This is approximated by checking that not all slots are filled
      AND: [
        {
          OR: [{ player1Id: null }, { player2Id: null }, { player3Id: null }, { player4Id: null }],
        },
      ],
    };

    // Apply optional filters
    if (filter.boardType) {
      where.boardType = filter.boardType;
    }
    if (filter.numPlayers) {
      where.maxPlayers = filter.numPlayers;
    }
    if (filter.fromDate) {
      where.endedAt = { ...(where.endedAt as object), gte: filter.fromDate };
    }
    if (filter.toDate) {
      where.endedAt = { ...(where.endedAt as object), lte: filter.toDate };
    }

    return where;
  }

  /**
   * Count human vs AI games matching the filter.
   */
  async countHumanVsAiGames(filter: GameRecordFilter = {}): Promise<number> {
    const db = this.getDb();
    const where = this.buildHumanVsAiWhereClause(filter);
    return db.game.count({ where });
  }

  /**
   * Export human vs AI games as a JSONL async generator.
   * Each yielded line is a JSON string representing a complete game record.
   */
  async *exportHumanVsAiGamesAsJsonl(
    filter: GameRecordFilter = {}
  ): AsyncGenerator<string, void, unknown> {
    const db = this.getDb();
    const where = this.buildHumanVsAiWhereClause(filter);
    const limit = filter.limit ?? 500;

    const games = await db.game.findMany({
      where,
      orderBy: { endedAt: 'asc' },
      take: limit,
      include: {
        moves: {
          orderBy: { moveNumber: 'asc' },
          include: { player: { select: { username: true } } },
        },
        player1: { select: { username: true, rating: true } },
        player2: { select: { username: true, rating: true } },
        player3: { select: { username: true, rating: true } },
        player4: { select: { username: true, rating: true } },
        winner: { select: { username: true } },
      },
    });

    for (const game of games) {
      if (!isCompletedGameRecord(game)) {
        continue;
      }

      try {
        const record = this.gameToGameRecord(game as GameWithRelations);
        if (record) {
          yield gameRecordToJsonlLine(record);
        }
      } catch (error) {
        logger.warn(`[GameRecordRepository] Failed to convert game ${game.id} to record: ${error}`);
      }
    }
  }

  /**
   * Get statistics about human vs AI games.
   */
  async getHumanVsAiStats(fromDate?: Date): Promise<{
    totalGames: number;
    humanWins: number;
    aiWins: number;
    draws: number;
    byBoardType: Record<string, number>;
    latestGameAt: Date | null;
  }> {
    const db = this.getDb();
    const filter: GameRecordFilter = fromDate ? { fromDate } : {};
    const where = this.buildHumanVsAiWhereClause(filter);

    // Get total count
    const totalGames = await db.game.count({ where });

    // Get counts by outcome
    // Human wins = winner is not null (a human player won)
    const humanWins = await db.game.count({
      where: { ...where, winnerId: { not: null } },
    });

    // AI wins = game completed but winner is null (AI won)
    const aiWins = await db.game.count({
      where: { ...where, winnerId: null, outcome: 'elimination' },
    });

    // Draws
    const draws = await db.game.count({
      where: { ...where, outcome: 'draw' },
    });

    // Get counts by board type
    const boardTypeCounts = await db.game.groupBy({
      by: ['boardType'],
      where,
      _count: true,
    });
    const byBoardType: Record<string, number> = {};
    for (const item of boardTypeCounts) {
      byBoardType[item.boardType] = item._count;
    }

    // Get latest game timestamp
    const latestGame = await db.game.findFirst({
      where,
      orderBy: { endedAt: 'desc' },
      select: { endedAt: true },
    });

    return {
      totalGames,
      humanWins,
      aiWins,
      draws,
      byBoardType,
      latestGameAt: latestGame?.endedAt ?? null,
    };
  }
}

// Singleton export
export const gameRecordRepository = new GameRecordRepository();
