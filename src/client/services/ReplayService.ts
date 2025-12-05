/**
 * ReplayService - HTTP client for the GameReplayDB REST API.
 *
 * This service communicates with the Python AI service's replay endpoints
 * to browse, query, and replay games stored in the SQLite database.
 *
 * Usage:
 *   const replayService = new ReplayService('http://localhost:8001');
 *   const games = await replayService.listGames({ board_type: 'square8' });
 *   const state = await replayService.getStateAtMove(gameId, 10);
 *
 * See: docs/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md
 */

import { readEnv } from '../../shared/utils/envFlags';
import type {
  ReplayGameListResponse,
  ReplayGameMetadata,
  ReplayGameQueryParams,
  ReplayMovesResponse,
  ReplayStateResponse,
  ReplayChoicesResponse,
  ReplayStatsResponse,
  StoreGameRequest,
  StoreGameResponse,
} from '../types/replay';
import type { GameHistoryResponse, GameHistoryMove, GameDetailsResponse } from './api';
import type {
  GameRecord,
  MoveRecord,
  GameRecordMetadata,
  FinalScore,
  GameOutcome,
  PlayerRecordInfo,
} from '../../shared/types/gameRecord';
import type {
  GameState,
  Move,
  Position,
  BoardType,
  LineInfo,
  Territory,
} from '../../shared/types/game';

/**
 * Get the AI service URL from environment or use default.
 *
 * Priority:
 * 1. RINGRIFT_AI_SERVICE_URL (runtime env)
 * 2. Default localhost:8001 for development
 */
function getAIServiceUrl(): string {
  const envUrl = readEnv('RINGRIFT_AI_SERVICE_URL');
  if (envUrl && typeof envUrl === 'string') {
    return envUrl.replace(/\/$/, '');
  }

  // Default for local development
  return 'http://localhost:8001';
}

/**
 * ReplayService provides access to the GameReplayDB REST API.
 */
export class ReplayService {
  private baseUrl: string;

  constructor(aiServiceUrl?: string) {
    this.baseUrl = `${aiServiceUrl ?? getAIServiceUrl()}/api/replay`;
  }

  /**
   * List games with optional filters.
   */
  async listGames(filters: ReplayGameQueryParams = {}): Promise<ReplayGameListResponse> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        params.set(key, String(value));
      }
    });

    const url = `${this.baseUrl}/games${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to list games: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get detailed metadata for a specific game including player info.
   */
  async getGame(gameId: string): Promise<ReplayGameMetadata> {
    const response = await fetch(`${this.baseUrl}/games/${encodeURIComponent(gameId)}`);

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get game: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get reconstructed game state at a specific move.
   *
   * @param gameId - Game identifier
   * @param moveNumber - Move number (0 = initial state, N = after move N)
   */
  async getStateAtMove(gameId: string, moveNumber: number): Promise<ReplayStateResponse> {
    const params = new URLSearchParams({ move_number: String(moveNumber) });
    const response = await fetch(
      `${this.baseUrl}/games/${encodeURIComponent(gameId)}/state?${params.toString()}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get state: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get moves for a game in a range.
   *
   * @param gameId - Game identifier
   * @param start - Start move number (inclusive, default 0)
   * @param end - End move number (exclusive, default all)
   * @param limit - Max moves to return (default 100)
   */
  async getMoves(
    gameId: string,
    start = 0,
    end?: number,
    limit = 100
  ): Promise<ReplayMovesResponse> {
    const params = new URLSearchParams({
      start: String(start),
      limit: String(limit),
    });
    if (end !== undefined) {
      params.set('end', String(end));
    }

    const response = await fetch(
      `${this.baseUrl}/games/${encodeURIComponent(gameId)}/moves?${params.toString()}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get moves: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get player choices made at a specific move.
   */
  async getChoices(gameId: string, moveNumber: number): Promise<ReplayChoicesResponse> {
    const params = new URLSearchParams({ move_number: String(moveNumber) });
    const response = await fetch(
      `${this.baseUrl}/games/${encodeURIComponent(gameId)}/choices?${params.toString()}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get choices: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get database statistics.
   */
  async getStats(): Promise<ReplayStatsResponse> {
    const response = await fetch(`${this.baseUrl}/stats`);

    if (!response.ok) {
      throw new Error(`Failed to get stats: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Store a game from the sandbox.
   *
   * Used by the sandbox UI to persist AI vs AI games to the database.
   */
  async storeGame(request: StoreGameRequest): Promise<StoreGameResponse> {
    const response = await fetch(`${this.baseUrl}/games`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to store game: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Check if the replay service is available.
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/stats`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

/**
 * Singleton instance for convenience.
 * Use this when you don't need to customize the AI service URL.
 */
let defaultInstance: ReplayService | null = null;

export function getReplayService(): ReplayService {
  if (!defaultInstance) {
    defaultInstance = new ReplayService();
  }
  return defaultInstance;
}

/**
 * Reset the singleton instance (for testing).
 */
export function resetReplayService(): void {
  defaultInstance = null;
}

// ──────────────────────────────────────────────────────────────────────────────
// Backend history → GameRecord adapter
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Narrow unknown to Position.
 */
function isPosition(value: unknown): value is Position {
  if (!value || typeof value !== 'object') return false;
  const obj = value as Record<string, unknown>;
  return typeof obj.x === 'number' && typeof obj.y === 'number';
}

/**
 * Normalise raw boardType string from the backend into a shared BoardType.
 * Falls back to 'square8' for unknown values.
 */
function toBoardType(raw: string): BoardType {
  if (raw === 'square8' || raw === 'square19' || raw === 'hexagonal') {
    return raw;
  }
  return 'square8';
}

/**
 * Map a GameHistoryResponse.result into a GameOutcome for GameRecord.
 * Defaults to 'abandonment' for unknown or missing reasons so that
 * downstream tooling always sees a valid outcome.
 */
function toGameOutcome(result: GameHistoryResponse['result'] | undefined): GameOutcome {
  if (!result) {
    return 'abandonment';
  }

  const reason = result.reason as string;

  // Structural stalemate / generic completion reason. For downstream training
  // and stats we treat this as a draw, mirroring GameSession._mapResultToOutcome.
  if (reason === 'game_completed') {
    return 'draw';
  }
  const allowed: GameOutcome[] = [
    'ring_elimination',
    'territory_control',
    'last_player_standing',
    'timeout',
    'resignation',
    'draw',
    'abandonment',
  ];

  if ((allowed as string[]).includes(reason)) {
    return reason as GameOutcome;
  }

  return 'abandonment';
}

/**
 * Build a minimal but valid FinalScore structure with zeroed fields for all
 * players. Backend history does not currently expose full score breakdown,
 * but replayHelpers only need structural correctness here.
 */
function createEmptyFinalScore(numPlayers: number): FinalScore {
  const ringsEliminated: FinalScore['ringsEliminated'] = {};
  const territorySpaces: FinalScore['territorySpaces'] = {};
  const ringsRemaining: FinalScore['ringsRemaining'] = {};

  for (let p = 1; p <= numPlayers; p += 1) {
    ringsEliminated[p] = 0;
    territorySpaces[p] = 0;
    ringsRemaining[p] = 0;
  }

  return { ringsEliminated, territorySpaces, ringsRemaining };
}

/**
 * Adapt a backend GameHistoryResponse + GameDetailsResponse pair into:
 * - A minimal GameRecord suitable for replayHelpers.reconstructStateAtMove.
 * - A parallel Move[] array for feeding MoveHistory / playback UIs.
 *
 * This keeps the mapping logic in one place and avoids leaking GameRecord
 * details throughout the UI.
 */
export function adaptHistoryToGameRecord(
  history: GameHistoryResponse,
  details: GameDetailsResponse
): { record: GameRecord; movesForDisplay: Move[] } {
  const boardType = toBoardType(details.boardType);
  const historyMoves = history.moves ?? [];

  // Map backend player IDs to seat indices for robust playerNumber mapping.
  const playerIdToSeat = new Map<string, number>();
  details.players.forEach((p, index) => {
    if (p && p.id) {
      playerIdToSeat.set(p.id, index + 1);
    }
  });

  // Infer number of players from details first, then from history payload.
  const distinctSeatsFromHistory = new Set<number>();
  historyMoves.forEach((entry) => {
    const raw = entry.moveData ?? {};
    const seat =
      typeof (raw as any).player === 'number' ? ((raw as any).player as number) : undefined;
    if (seat) distinctSeatsFromHistory.add(seat);
  });

  const numPlayers =
    details.players.length > 0 ? details.players.length : distinctSeatsFromHistory.size || 2;

  // Build PlayerRecordInfo array from game details, stubbing unknown seats.
  const players: PlayerRecordInfo[] = [];
  for (let i = 0; i < numPlayers; i += 1) {
    const seatIndex = i;
    const p = details.players[seatIndex];
    players.push({
      playerNumber: i + 1,
      username: p?.username ?? `Player ${i + 1}`,
      playerType: 'human',
      ...(typeof p?.rating === 'number' ? { ratingBefore: p.rating } : {}),
    });
  }

  // Helper to choose a player seat for a history entry.
  const inferPlayerSeat = (entry: GameHistoryMove): number => {
    const raw = entry.moveData ?? {};
    const seatFromMove =
      typeof (raw as any).player === 'number' ? ((raw as any).player as number) : undefined;
    if (seatFromMove && seatFromMove >= 1 && seatFromMove <= numPlayers) {
      return seatFromMove;
    }
    const seatFromId = playerIdToSeat.get(entry.playerId);
    if (seatFromId && seatFromId >= 1 && seatFromId <= numPlayers) {
      return seatFromId;
    }
    return 1;
  };

  const moveRecords: MoveRecord[] = [];
  const movesForDisplay: Move[] = [];

  historyMoves.forEach((entry) => {
    const raw = entry.moveData ?? {};
    const type = ((raw as any).type ?? entry.moveType) as MoveRecord['type'];

    const from = isPosition((raw as any).from) ? ((raw as any).from as Position) : undefined;
    const to = isPosition((raw as any).to) ? ((raw as any).to as Position) : undefined;
    const captureTarget = isPosition((raw as any).captureTarget)
      ? ((raw as any).captureTarget as Position)
      : undefined;

    const placementCount =
      typeof (raw as any).placementCount === 'number'
        ? ((raw as any).placementCount as number)
        : undefined;
    const placedOnStack =
      typeof (raw as any).placedOnStack === 'boolean'
        ? ((raw as any).placedOnStack as boolean)
        : undefined;

    const formedLines =
      Array.isArray((raw as any).formedLines) && (raw as any).formedLines.length > 0
        ? ((raw as any).formedLines as LineInfo[])
        : undefined;
    const collapsedMarkers =
      Array.isArray((raw as any).collapsedMarkers) && (raw as any).collapsedMarkers.length > 0
        ? ((raw as any).collapsedMarkers as Position[])
        : undefined;
    const disconnectedRegions =
      Array.isArray((raw as any).disconnectedRegions) && (raw as any).disconnectedRegions.length > 0
        ? ((raw as any).disconnectedRegions as Territory[])
        : undefined;
    const eliminatedRings =
      Array.isArray((raw as any).eliminatedRings) && (raw as any).eliminatedRings.length > 0
        ? ((raw as any).eliminatedRings as { player: number; count: number }[])
        : undefined;

    const thinkTimeCandidate = (raw as any).thinkTimeMs ?? (raw as any).thinkTime;
    const thinkTimeMs =
      typeof thinkTimeCandidate === 'number' && Number.isFinite(thinkTimeCandidate)
        ? (thinkTimeCandidate as number)
        : 0;

    const player = inferPlayerSeat(entry);

    const recordMove: MoveRecord = {
      moveNumber: entry.moveNumber,
      player,
      type,
      thinkTimeMs,
      ...(from ? { from } : {}),
      ...(to ? { to } : {}),
      ...(captureTarget ? { captureTarget } : {}),
      ...(placementCount !== undefined ? { placementCount } : {}),
      ...(placedOnStack !== undefined ? { placedOnStack } : {}),
      ...(formedLines ? { formedLines } : {}),
      ...(collapsedMarkers ? { collapsedMarkers } : {}),
      ...(disconnectedRegions ? { disconnectedRegions } : {}),
      ...(eliminatedRings ? { eliminatedRings } : {}),
    };

    moveRecords.push(recordMove);

    const uiMove: Move = {
      id: `history-${history.gameId}-${entry.moveNumber}`,
      type,
      player,
      ...(from ? { from } : {}),
      to: to ?? from ?? { x: 0, y: 0 },
      ...(captureTarget ? { captureTarget } : {}),
      ...(formedLines ? { formedLines } : {}),
      ...(collapsedMarkers ? { collapsedMarkers } : {}),
      ...(disconnectedRegions ? { disconnectedRegions } : {}),
      ...(eliminatedRings ? { eliminatedRings } : {}),
      timestamp: new Date(entry.timestamp),
      thinkTime: thinkTimeMs,
      moveNumber: entry.moveNumber,
    };

    movesForDisplay.push(uiMove);
  });

  const firstHistoryTs = historyMoves[0]?.timestamp;
  const lastHistoryTs = historyMoves[historyMoves.length - 1]?.timestamp ?? firstHistoryTs;

  const startedAt = details.startedAt ?? firstHistoryTs ?? new Date().toISOString();
  const endedAt = details.endedAt ?? lastHistoryTs ?? startedAt;

  const totalDurationMs = Math.max(0, new Date(endedAt).getTime() - new Date(startedAt).getTime());
  const outcome = toGameOutcome(history.result);
  const finalScore = createEmptyFinalScore(numPlayers);

  const record: GameRecord = {
    id: history.gameId,
    boardType,
    numPlayers,
    isRated: details.isRated,
    players,
    winner: typeof history.result?.winner === 'number' ? history.result.winner : undefined,
    outcome,
    finalScore,
    startedAt,
    endedAt,
    totalMoves: history.totalMoves,
    totalDurationMs,
    moves: moveRecords,
    metadata: {
      recordVersion: '1.0.0-client-replay',
      createdAt: endedAt,
      source: 'online_game',
      sourceId: history.gameId,
      tags: ['client_replay', 'backend_history'],
    } as GameRecordMetadata,
  };

  return { record, movesForDisplay };
}
