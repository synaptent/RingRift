/**
 * Service for accessing recorded self-play games from SQLite databases.
 *
 * This service provides read-only access to games stored by the Python
 * ai-service during CMA-ES training, self-play soaks, and other training runs.
 */

import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { gunzipSync } from 'zlib';
import type { GameStatus as PrismaGameStatus } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';
import { GamePersistenceService } from './GamePersistenceService';
import { gameRecordRepository } from './GameRecordRepository';
import { getOrCreateAIUser } from './AIUserService';
import type { BoardType as DomainBoardType, GameState, Move } from '../../shared/types/game';
import type { FinalScore, GameOutcome, RecordSource } from '../../shared/types/gameRecord';
import { processTurn } from '../../shared/engine/orchestration/turnOrchestrator';

// Types for game data
export interface SelfPlayGameSummary {
  gameId: string;
  boardType: string;
  numPlayers: number;
  winner: number | null;
  totalMoves: number;
  totalTurns: number;
  createdAt: string;
  completedAt: string | null;
  source: string | null;
  terminationReason: string | null;
  durationMs: number | null;
}

export interface SelfPlayGameDetail extends SelfPlayGameSummary {
  initialState: unknown;
  moves: SelfPlayMove[];
  players: SelfPlayPlayer[];
}

export interface SelfPlayMove {
  moveNumber: number;
  turnNumber: number;
  player: number;
  phase: string;
  moveType: string;
  move: unknown;
  thinkTimeMs: number | null;
  engineEval: number | null;
}

export interface SelfPlayPlayer {
  playerNumber: number;
  playerType: string;
  aiType: string | null;
  aiDifficulty: number | null;
  aiProfileId: string | null;
  finalEliminatedRings: number | null;
  finalTerritorySpaces: number | null;
}

export interface GameListOptions {
  boardType?: string | undefined;
  numPlayers?: number | undefined;
  source?: string | undefined;
  hasWinner?: boolean | undefined;
  limit?: number | undefined;
  offset?: number | undefined;
}

export interface DatabaseInfo {
  path: string;
  name: string;
  gameCount: number;
  createdAt: string | null;
}

/**
 * Service for querying self-play game databases.
 */
export class SelfPlayGameService {
  private dbCache: Map<string, Database.Database> = new Map();

  /**
   * Get or open a database connection.
   */
  private getDb(dbPath: string): Database.Database {
    if (!fs.existsSync(dbPath)) {
      throw new Error(`Database not found: ${dbPath}`);
    }

    let db = this.dbCache.get(dbPath);
    if (!db) {
      db = new Database(dbPath, { readonly: true });
      this.dbCache.set(dbPath, db);
    }
    return db;
  }

  /**
   * Close all open database connections.
   */
  closeAll(): void {
    for (const db of this.dbCache.values()) {
      db.close();
    }
    this.dbCache.clear();
  }

  /**
   * List available game databases in a directory.
   */
  listDatabases(rootDir: string): DatabaseInfo[] {
    const results: DatabaseInfo[] = [];

    const searchPaths = [
      path.join(rootDir, 'data', 'games'),
      path.join(rootDir, 'ai-service', 'logs', 'cmaes'),
      path.join(rootDir, 'ai-service', 'data', 'games'),
    ];

    for (const searchPath of searchPaths) {
      if (!fs.existsSync(searchPath)) continue;

      this.findDatabasesRecursive(searchPath, results, 7);
    }

    return results;
  }

  private findDatabasesRecursive(dir: string, results: DatabaseInfo[], maxDepth: number): void {
    if (maxDepth <= 0) return;

    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          this.findDatabasesRecursive(fullPath, results, maxDepth - 1);
        } else if (entry.name === 'games.db' || entry.name.endsWith('.db')) {
          try {
            const db = this.getDb(fullPath);
            const countResult = db.prepare('SELECT COUNT(*) as count FROM games').get() as {
              count: number;
            };

            // Skip empty databases so the sandbox self-play browser only shows
            // databases that contain at least one recorded game. This keeps the
            // dropdown tidy when old runs created placeholder DBs with no data.
            if (!countResult.count || countResult.count <= 0) {
              continue;
            }

            const firstGame = db
              .prepare('SELECT created_at FROM games ORDER BY created_at LIMIT 1')
              .get() as
              | {
                  created_at: string;
                }
              | undefined;

            results.push({
              path: fullPath,
              name: path.relative(process.cwd(), fullPath),
              gameCount: countResult.count,
              createdAt: firstGame?.created_at ?? null,
            });
          } catch {
            // Skip invalid databases
          }
        }
      }
    } catch {
      // Skip directories we can't read
    }
  }

  /**
   * List games from a specific database.
   */
  listGames(dbPath: string, options: GameListOptions = {}): SelfPlayGameSummary[] {
    const db = this.getDb(dbPath);

    let sql = `
      SELECT
        game_id as gameId,
        board_type as boardType,
        num_players as numPlayers,
        winner,
        total_moves as totalMoves,
        total_turns as totalTurns,
        created_at as createdAt,
        completed_at as completedAt,
        source,
        termination_reason as terminationReason,
        duration_ms as durationMs
      FROM games
      WHERE 1=1
    `;
    const params: unknown[] = [];

    if (options.boardType) {
      sql += ' AND board_type = ?';
      params.push(options.boardType);
    }

    if (options.numPlayers !== undefined) {
      sql += ' AND num_players = ?';
      params.push(options.numPlayers);
    }

    if (options.source) {
      sql += ' AND source = ?';
      params.push(options.source);
    }

    if (options.hasWinner !== undefined) {
      sql += options.hasWinner ? ' AND winner IS NOT NULL' : ' AND winner IS NULL';
    }

    sql += ' ORDER BY created_at DESC';

    if (options.limit) {
      sql += ' LIMIT ?';
      params.push(options.limit);
    }

    if (options.offset) {
      sql += ' OFFSET ?';
      params.push(options.offset);
    }

    return db.prepare(sql).all(...params) as SelfPlayGameSummary[];
  }

  /**
   * Get a single game with full details.
   */
  getGame(dbPath: string, gameId: string): SelfPlayGameDetail | null {
    const db = this.getDb(dbPath);

    // Get game summary
    const game = db
      .prepare(
        `
      SELECT
        game_id as gameId,
        board_type as boardType,
        num_players as numPlayers,
        winner,
        total_moves as totalMoves,
        total_turns as totalTurns,
        created_at as createdAt,
        completed_at as completedAt,
        source,
        termination_reason as terminationReason,
        duration_ms as durationMs
      FROM games
      WHERE game_id = ?
    `
      )
      .get(gameId) as SelfPlayGameSummary | undefined;

    if (!game) return null;

    // Get initial state
    const stateRow = db
      .prepare(
        `
      SELECT initial_state_json, compressed
      FROM game_initial_state
      WHERE game_id = ?
    `
      )
      .get(gameId) as { initial_state_json: string | Buffer; compressed: number } | undefined;

    let initialState: unknown = null;
    if (stateRow) {
      let stateJson = stateRow.initial_state_json;
      if (stateRow.compressed && Buffer.isBuffer(stateJson)) {
        stateJson = gunzipSync(stateJson).toString('utf-8');
      }
      initialState = JSON.parse(typeof stateJson === 'string' ? stateJson : stateJson.toString());
    }

    // Get moves
    const moves = db
      .prepare(
        `
      SELECT
        move_number as moveNumber,
        turn_number as turnNumber,
        player,
        phase,
        move_type as moveType,
        move_json as moveJson,
        think_time_ms as thinkTimeMs,
        engine_eval as engineEval
      FROM game_moves
      WHERE game_id = ?
      ORDER BY move_number
    `
      )
      .all(gameId) as Array<{
      moveNumber: number;
      turnNumber: number;
      player: number;
      phase: string;
      moveType: string;
      moveJson: string;
      thinkTimeMs: number | null;
      engineEval: number | null;
    }>;

    const parsedMoves: SelfPlayMove[] = moves.map((m) => ({
      moveNumber: m.moveNumber,
      turnNumber: m.turnNumber,
      player: m.player,
      phase: m.phase,
      moveType: m.moveType,
      move: JSON.parse(m.moveJson),
      thinkTimeMs: m.thinkTimeMs,
      engineEval: m.engineEval,
    }));

    // Get players
    const players = db
      .prepare(
        `
      SELECT
        player_number as playerNumber,
        player_type as playerType,
        ai_type as aiType,
        ai_difficulty as aiDifficulty,
        ai_profile_id as aiProfileId,
        final_eliminated_rings as finalEliminatedRings,
        final_territory_spaces as finalTerritorySpaces
      FROM game_players
      WHERE game_id = ?
      ORDER BY player_number
    `
      )
      .all(gameId) as SelfPlayPlayer[];

    return {
      ...game,
      initialState,
      moves: parsedMoves,
      players,
    };
  }

  /**
   * Get game state at a specific move number.
   * Uses snapshots if available, otherwise reconstructs from initial state by replaying moves.
   */
  getStateAtMove(dbPath: string, gameId: string, moveNumber: number): GameState | null {
    const db = this.getDb(dbPath);

    // Try to find a snapshot at or before the target move
    const snapshot = db
      .prepare(
        `
      SELECT state_json, compressed, move_number
      FROM game_state_snapshots
      WHERE game_id = ? AND move_number <= ?
      ORDER BY move_number DESC
      LIMIT 1
    `
      )
      .get(gameId, moveNumber) as
      | {
          state_json: string | Buffer;
          compressed: number;
          move_number: number;
        }
      | undefined;

    let baseState: GameState;
    let startMoveNumber: number;

    if (snapshot) {
      let stateJson = snapshot.state_json;
      if (snapshot.compressed && Buffer.isBuffer(stateJson)) {
        stateJson = gunzipSync(stateJson).toString('utf-8');
      }
      baseState = JSON.parse(
        typeof stateJson === 'string' ? stateJson : stateJson.toString()
      ) as GameState;
      startMoveNumber = snapshot.move_number;

      // If snapshot is at exact target move, return it directly
      if (startMoveNumber === moveNumber) {
        return baseState;
      }
    } else {
      // No snapshot - start from initial state
      const game = this.getGame(dbPath, gameId);
      if (!game?.initialState) {
        return null;
      }
      baseState = game.initialState as GameState;
      startMoveNumber = 0;
    }

    // Return initial state directly if target is move 0
    if (moveNumber === 0) {
      return baseState;
    }

    // Get moves from startMoveNumber + 1 to moveNumber
    const moves = db
      .prepare(
        `
      SELECT move_json
      FROM game_moves
      WHERE game_id = ? AND move_number > ? AND move_number <= ?
      ORDER BY move_number
    `
      )
      .all(gameId, startMoveNumber, moveNumber) as Array<{ move_json: string }>;

    // Apply each move to reconstruct state
    let currentState = baseState;
    for (const row of moves) {
      const move = JSON.parse(row.move_json) as Move;
      try {
        const result = processTurn(currentState, move, { replayCompatibility: true });
        currentState = result.nextState;
      } catch (error) {
        logger.warn('Failed to apply move during state reconstruction', {
          gameId,
          moveNumber,
          error: error instanceof Error ? error.message : String(error),
        });
        return null;
      }
    }

    return currentState;
  }

  /**
   * Get aggregate statistics for a database.
   */
  getStats(dbPath: string): {
    totalGames: number;
    byBoardType: Record<string, number>;
    byNumPlayers: Record<number, number>;
    byWinner: Record<string, number>;
    avgMoves: number;
    avgDuration: number | null;
  } {
    const db = this.getDb(dbPath);

    const total = db.prepare('SELECT COUNT(*) as count FROM games').get() as { count: number };

    const byBoard = db
      .prepare(
        `
      SELECT board_type, COUNT(*) as count
      FROM games
      GROUP BY board_type
    `
      )
      .all() as Array<{ board_type: string; count: number }>;

    const byPlayers = db
      .prepare(
        `
      SELECT num_players, COUNT(*) as count
      FROM games
      GROUP BY num_players
    `
      )
      .all() as Array<{ num_players: number; count: number }>;

    const byWinner = db
      .prepare(
        `
      SELECT
        CASE WHEN winner IS NULL THEN 'draw' ELSE 'p' || winner END as winner_key,
        COUNT(*) as count
      FROM games
      GROUP BY winner_key
    `
      )
      .all() as Array<{ winner_key: string; count: number }>;

    const avgStats = db
      .prepare(
        `
      SELECT
        AVG(total_moves) as avgMoves,
        AVG(duration_ms) as avgDuration
      FROM games
    `
      )
      .get() as { avgMoves: number; avgDuration: number | null };

    return {
      totalGames: total.count,
      byBoardType: Object.fromEntries(byBoard.map((r) => [r.board_type, r.count])),
      byNumPlayers: Object.fromEntries(byPlayers.map((r) => [r.num_players, r.count])),
      byWinner: Object.fromEntries(byWinner.map((r) => [r.winner_key, r.count])),
      avgMoves: avgStats.avgMoves,
      avgDuration: avgStats.avgDuration,
    };
  }
}

/**
 * Options for importing a recorded self-play game into the canonical
 * Postgres Game + GameRecord pipeline.
 */
export interface SelfPlayImportOptions {
  dbPath: string;
  gameId: string;
  source?: RecordSource;
  tags?: string[];
}

/**
 * Map a self-play terminationReason string into the shared GameOutcome union.
 * Falls back to last_player_standing/draw when the reason is unknown.
 */
function mapSelfPlayTerminationToOutcome(
  terminationReason: string | null,
  winner: number | null
): GameOutcome {
  switch (terminationReason) {
    case 'ring_elimination':
      return 'ring_elimination';
    case 'territory':
    case 'territory_control':
      return 'territory_control';
    case 'last_player_standing':
      return 'last_player_standing';
    case 'timeout':
    case 'max_turns_reached':
    case 'max_moves_reached':
      return 'timeout';
    case 'resignation':
      return 'resignation';
    case 'abandonment':
      return 'abandonment';
    case 'draw':
    case 'stalemate':
      return 'draw';
    default:
      if (winner !== null && winner !== undefined) {
        return 'last_player_standing';
      }
      return 'draw';
  }
}

/**
 * Build a FinalScore object from self-play player summary rows.
 */
function buildFinalScoreFromSelfPlayPlayers(players: SelfPlayPlayer[]): FinalScore {
  const ringsEliminated: Record<number, number> = {};
  const territorySpaces: Record<number, number> = {};
  const ringsRemaining: Record<number, number> = {};

  for (const player of players) {
    const seat = player.playerNumber;
    ringsEliminated[seat] = player.finalEliminatedRings ?? 0;
    territorySpaces[seat] = player.finalTerritorySpaces ?? 0;
    ringsRemaining[seat] = 0;
  }

  return { ringsEliminated, territorySpaces, ringsRemaining };
}

/**
 * Import a completed self-play game from a SQLite GameReplayDB into the
 * canonical Postgres Game + GameRecord pipeline.
 *
 * This function:
 * - Creates an unrated Game row with AI-only seats.
 * - Persists all recorded moves via GamePersistenceService.saveMoveSync so
 *   moveData JSON is available to GameRecordRepository.
 * - Computes a best-effort FinalScore and GameOutcome from self-play metadata.
 * - Writes finalState/finalScore/outcome/recordMetadata via
 *   GameRecordRepository.saveGameRecord so that JSONL exporters can stream it.
 *
 * Returns the newly created Game.id.
 */
export async function importSelfPlayGameAsGameRecord(
  options: SelfPlayImportOptions,
  serviceOverride?: Pick<SelfPlayGameService, 'getGame' | 'getStateAtMove'>
): Promise<string> {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw new Error('Database not available');
  }

  const service = serviceOverride ?? getSelfPlayGameService();
  const resolvedDbPath = path.isAbsolute(options.dbPath)
    ? options.dbPath
    : path.join(process.cwd(), options.dbPath);

  const detail = service.getGame(resolvedDbPath, options.gameId);
  if (!detail) {
    throw new Error(`Self-play game ${options.gameId} not found in DB ${resolvedDbPath}`);
  }

  if (!detail.completedAt) {
    throw new Error(`Self-play game ${options.gameId} is not completed and cannot be imported`);
  }

  const createdAt = new Date(detail.createdAt);
  const endedAt = new Date(detail.completedAt);

  let gameId: string;
  try {
    gameId = await GamePersistenceService.createGame({
      boardType: detail.boardType as DomainBoardType,
      maxPlayers: detail.numPlayers,
      // Self-play imports use a neutral, non-rated time control bucket. Since
      // TimeControl.type is restricted to the rated ladder buckets, we map
      // these offline games into the slowest category so downstream tooling
      // can treat them consistently without special-casing a new type.
      timeControl: { type: 'classical', initialTime: 0, increment: 0 },
      isRated: false,
      allowSpectators: true,
    });
  } catch (err) {
    logger.error('Failed to create Game row for self-play import', {
      dbPath: resolvedDbPath,
      selfPlayGameId: detail.gameId,
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }

  try {
    // Align basic timestamps and status with the recorded episode.
    await prisma.game.update({
      where: { id: gameId },
      data: {
        status: 'completed' as PrismaGameStatus,
        createdAt,
        startedAt: createdAt,
        endedAt,
        updatedAt: new Date(),
      },
    });
  } catch (err) {
    logger.warn('Failed to update Game timestamps for self-play import', {
      gameId,
      error: err instanceof Error ? err.message : String(err),
    });
  }

  // Persist moves using the shared persistence service so that moveData JSON
  // is available to GameRecordRepository.
  try {
    const aiUser = await getOrCreateAIUser();

    // Define a mutable move type for enrichment
    type MutableMove = Move & {
      moveNumber?: number;
      thinkTime?: number;
      timestamp?: Date | string;
    };

    for (const m of detail.moves) {
      const move = m.move as MutableMove;

      if (typeof move.moveNumber !== 'number') {
        move.moveNumber = m.moveNumber;
      }
      if (typeof move.thinkTime !== 'number') {
        move.thinkTime = m.thinkTimeMs ?? 0;
      }
      const rawTs: Date | string | undefined = move.timestamp;
      if (rawTs === undefined || typeof rawTs === 'string') {
        move.timestamp = typeof rawTs === 'string' ? new Date(rawTs) : createdAt;
      }
      if (typeof move.player !== 'number') {
        (move as Move & { player: number }).player = m.player;
      }

      await GamePersistenceService.saveMoveSync({
        gameId,
        playerId: aiUser.id,
        moveNumber: move.moveNumber as number,
        move: move as Move,
      });
    }
  } catch (err) {
    logger.error('Failed to persist self-play moves into Postgres', {
      gameId,
      dbPath: resolvedDbPath,
      selfPlayGameId: detail.gameId,
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }

  // Compute outcome, finalScore, and finalState for GameRecord storage.
  const finalScore = buildFinalScoreFromSelfPlayPlayers(detail.players);
  const outcome = mapSelfPlayTerminationToOutcome(detail.terminationReason, detail.winner);
  const finalStateRaw =
    service.getStateAtMove(resolvedDbPath, detail.gameId, detail.totalMoves) ??
    detail.initialState ??
    {};

  const finalState = finalStateRaw as GameState;

  const tagSet = new Set<string>(options.tags ?? []);
  tagSet.add('self_play');
  tagSet.add('import:selfplay_sqlite');
  tagSet.add(`db:${path.basename(resolvedDbPath)}`);
  tagSet.add(`selfplay_game_id:${detail.gameId}`);
  if (detail.source) {
    tagSet.add(`selfplay_source:${detail.source}`);
  }
  if (detail.terminationReason) {
    tagSet.add(`termination:${detail.terminationReason}`);
  }
  if (detail.numPlayers) {
    tagSet.add(`players:${detail.numPlayers}`);
  }
  if (detail.winner !== null && detail.winner !== undefined) {
    tagSet.add(`winner_seat:${detail.winner}`);
  }

  try {
    await gameRecordRepository.saveGameRecord(gameId, finalState, outcome, finalScore, {
      source: options.source ?? 'self_play',
      sourceId: detail.gameId,
      tags: Array.from(tagSet),
    });
  } catch (err) {
    logger.warn('Failed to save canonical GameRecord for self-play import', {
      gameId,
      dbPath: resolvedDbPath,
      selfPlayGameId: detail.gameId,
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }

  // After record storage, ensure endedAt reflects the original episode end time.
  try {
    await prisma.game.update({
      where: { id: gameId },
      data: {
        status: 'completed' as PrismaGameStatus,
        createdAt,
        startedAt: createdAt,
        endedAt,
        updatedAt: new Date(),
      },
    });
  } catch (err) {
    logger.warn('Failed to re-apply Game timestamps after GameRecord save', {
      gameId,
      error: err instanceof Error ? err.message : String(err),
    });
  }

  return gameId;
}

// Singleton instance
let serviceInstance: SelfPlayGameService | null = null;

export function getSelfPlayGameService(): SelfPlayGameService {
  if (!serviceInstance) {
    serviceInstance = new SelfPlayGameService();
  }
  return serviceInstance;
}
