/**
 * Unit tests for the GameRecord JSONL export pipeline.
 *
 * These tests validate:
 * - Mapping from DB rows (via GameRecordRepository.exportAsJsonl)
 *   into canonical GameRecord objects/JSONL lines.
 * - Basic CLI/filter helpers used by scripts/export-game-records-jsonl.ts.
 */

import path from 'path';
import { Prisma } from '@prisma/client';

import {
  gameRecordRepository,
  type GameRecordFilter,
} from '../../src/server/services/GameRecordRepository';
import { jsonlLineToGameRecord } from '../../src/shared/types/gameRecord';
import {
  buildFilterFromCliArgs,
  parseArgs,
  resolveOutputPath,
  type CliArgs,
} from '../../scripts/export-game-records-jsonl';
import * as SelfPlayService from '../../src/server/services/SelfPlayGameService';
import type { SelfPlayGameDetail } from '../../src/server/services/SelfPlayGameService';

// Mock database client used by GameRecordRepository and self-play import
// to avoid touching a real database.
const mockGameFindMany = jest.fn();
const mockGameCreate = jest.fn();
const mockGameUpdate = jest.fn();
const mockMoveCreate = jest.fn();
const mockUserFindUnique = jest.fn();
const mockUserCreate = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findMany: mockGameFindMany,
      create: mockGameCreate,
      update: mockGameUpdate,
    },
    move: {
      create: mockMoveCreate,
    },
    user: {
      findUnique: mockUserFindUnique,
      create: mockUserCreate,
    },
  }),
}));

// Mock logger to keep test output clean.
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('GameRecordRepository.exportAsJsonl mapping', () => {
  beforeEach(() => {
    mockGameFindMany.mockReset();
    mockGameCreate.mockReset();
    mockGameUpdate.mockReset();
    mockMoveCreate.mockReset();
    mockUserFindUnique.mockReset();
    mockUserCreate.mockReset();
  });

  it('maps a completed DB game row into a canonical GameRecord JSONL line', async () => {
    const createdAt = new Date('2024-01-15T10:00:00Z');
    const startedAt = new Date('2024-01-15T10:05:00Z');
    const endedAt = new Date('2024-01-15T10:10:00Z');

    // Minimal Game row with relations matching GameWithRelations shape.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const sampleGame: any = {
      id: 'game-1',
      boardType: 'square8',
      maxPlayers: 2,
      isRated: true,
      rngSeed: 123,
      status: 'completed',
      createdAt,
      startedAt,
      endedAt,
      finalState: { some: 'state' },
      finalScore: {
        ringsEliminated: { 1: 5, 2: 3 },
        territorySpaces: { 1: 8, 2: 4 },
        ringsRemaining: { 1: 13, 2: 15 },
      },
      outcome: 'ring_elimination',
      recordMetadata: {
        recordVersion: '1.0.0',
        createdAt: endedAt.toISOString(),
        source: 'online_game',
        tags: ['test'],
      },
      moves: [
        {
          id: 'm1',
          moveNumber: 1,
          moveType: 'place_ring',
          moveData: {
            player: 1,
            type: 'place_ring',
            to: { x: 3, y: 3 },
            placementCount: 1,
            thinkTime: 500,
          },
          timestamp: new Date('2024-01-15T10:05:30Z'),
          player: { username: 'Alice' },
        },
        {
          id: 'm2',
          moveNumber: 2,
          moveType: 'move_stack',
          moveData: {
            player: 2,
            type: 'move_stack',
            from: { x: 3, y: 3 },
            to: { x: 4, y: 4 },
            thinkTime: 700,
            captureTarget: { x: 4, y: 4 },
            formedLines: [
              {
                positions: [
                  { x: 4, y: 4 },
                  { x: 5, y: 5 },
                ],
              },
            ],
            collapsedMarkers: [{ x: 4, y: 4 }],
          },
          timestamp: new Date('2024-01-15T10:06:00Z'),
          player: { username: 'Bob' },
        },
      ],
      player1: { username: 'Alice', rating: 1600 },
      player2: { username: 'Bob', rating: 1500 },
      player3: null,
      player4: null,
      winner: { username: 'Alice' },
    };

    let lastWhere: unknown;
    let lastTake: number | undefined;
    let lastSkip: number | undefined;

    mockGameFindMany.mockImplementation(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ({ where, take, skip }: any) => {
        lastWhere = where;
        lastTake = take;
        lastSkip = skip;

        // Single page of results followed by empty page.
        if (skip && skip > 0) {
          return [];
        }
        return [sampleGame];
      }
    );

    const filter: GameRecordFilter = {
      // BoardType is a string union under the hood; using string literal here is sufficient.
      boardType: 'square8' as any,
      isRated: true,
      limit: 1,
    };

    const lines: string[] = [];
    for await (const line of gameRecordRepository.exportAsJsonl(filter)) {
      lines.push(line);
    }

    expect(lines).toHaveLength(1);

    const record = jsonlLineToGameRecord(lines[0]);

    // Top-level fields
    expect(record.id).toBe('game-1');
    expect(record.boardType).toBe('square8');
    expect(record.numPlayers).toBe(2);
    expect(record.isRated).toBe(true);
    expect(record.rngSeed).toBe(123);

    // Players
    expect(record.players).toHaveLength(2);
    expect(record.players[0]).toMatchObject({
      playerNumber: 1,
      username: 'Alice',
      playerType: 'human',
      ratingBefore: 1600,
    });
    expect(record.players[1]).toMatchObject({
      playerNumber: 2,
      username: 'Bob',
      playerType: 'human',
      ratingBefore: 1500,
    });

    // Outcome & score
    expect(record.winner).toBe(1);
    expect(record.outcome).toBe('ring_elimination');
    expect(record.finalScore.ringsEliminated[1]).toBe(5);
    expect(record.finalScore.ringsEliminated[2]).toBe(3);

    // Timing
    expect(record.totalMoves).toBe(2);
    expect(record.totalDurationMs).toBeGreaterThan(0);

    // Moves – ensure mapping from moveData -> MoveRecord is conservative and structured.
    expect(record.moves).toHaveLength(2);
    expect(record.moves[0]).toMatchObject({
      moveNumber: 1,
      player: 1,
      type: 'place_ring',
      thinkTimeMs: 500,
      to: { x: 3, y: 3 },
      placementCount: 1,
    });
    expect(record.moves[1]).toMatchObject({
      moveNumber: 2,
      player: 2,
      type: 'move_stack',
      thinkTimeMs: 700,
      from: { x: 3, y: 3 },
      to: { x: 4, y: 4 },
      captureTarget: { x: 4, y: 4 },
    });

    // Metadata
    expect(record.metadata.source).toBe('online_game');
    expect(record.metadata.tags).toEqual(['test']);

    // Underlying Prisma filter & pagination honour the GameRecordFilter.
    expect(lastWhere).toMatchObject({
      boardType: 'square8',
      isRated: true,
      finalState: { not: Prisma.DbNull },
      outcome: { not: null },
    });
    expect(lastTake).toBe(1);
    expect(lastSkip).toBe(0);
  });
});

describe('export-game-records-jsonl CLI helpers', () => {
  it('parses CLI arguments into CliArgs and builds a matching GameRecordFilter', () => {
    const sinceIso = '2024-01-01T00:00:00Z';
    const untilIso = '2024-02-01T00:00:00Z';

    const argv = [
      'node',
      'export-game-records-jsonl.ts',
      '--output',
      'custom.jsonl',
      '--board-type',
      'square8',
      '--limit',
      '10',
      '--rated-only',
      '--since',
      sinceIso,
      '--until',
      untilIso,
    ];

    const args = parseArgs(argv);
    expect(args).not.toBeNull();

    const nonNullArgs = args as CliArgs;
    expect(nonNullArgs.output).toBe('custom.jsonl');
    expect(nonNullArgs.boardType).toBe('square8');
    expect(nonNullArgs.limit).toBe(10);
    expect(nonNullArgs.ratedOnly).toBe(true);

    const expectedSince = new Date(sinceIso);
    const expectedUntil = new Date(untilIso);

    expect(nonNullArgs.since?.getTime()).toBe(expectedSince.getTime());
    expect(nonNullArgs.until?.getTime()).toBe(expectedUntil.getTime());

    const filter = buildFilterFromCliArgs(nonNullArgs);
    expect(filter.boardType).toBe('square8');
    expect(filter.isRated).toBe(true);
    expect(filter.limit).toBe(10);
    expect(filter.fromDate?.getTime()).toBe(expectedSince.getTime());
    expect(filter.toDate?.getTime()).toBe(expectedUntil.getTime());
  });

  it('resolves a default output path when --output is omitted', () => {
    const resolved = resolveOutputPath(undefined);

    // Should be absolute and end with the expected directory & extension.
    expect(path.isAbsolute(resolved)).toBe(true);
    expect(resolved).toContain(path.join('results', 'game-records'));
    expect(resolved.endsWith('.jsonl')).toBe(true);
  });
});

describe('SelfPlayGameService.importSelfPlayGameAsGameRecord', () => {
  it('imports a self-play game and saves a GameRecord with source "self_play"', async () => {
    const createdAtIso = '2024-01-01T00:00:00Z';
    const completedAtIso = '2024-01-01T00:05:00Z';

    const detail: SelfPlayGameDetail = {
      gameId: 'sp-1',
      boardType: 'square8',
      numPlayers: 2,
      winner: 1,
      totalMoves: 1,
      totalTurns: 1,
      createdAt: createdAtIso,
      completedAt: completedAtIso,
      source: 'self_play_pipeline',
      terminationReason: 'ring_elimination',
      durationMs: 5 * 60 * 1000,
      initialState: {},
      moves: [
        {
          moveNumber: 1,
          turnNumber: 1,
          player: 1,
          phase: 'ring_placement',
          moveType: 'place_ring',
          // Minimal canonical Move payload; SelfPlay import will normalise
          // timestamps, thinkTime, moveNumber, and player.
          move: {
            id: 'm1',
            type: 'place_ring',
            player: 1,
            to: { x: 3, y: 3 },
            timestamp: new Date(createdAtIso),
            thinkTime: 100,
            moveNumber: 1,
          } as any,
          thinkTimeMs: 100,
          engineEval: null,
        },
      ],
      players: [
        {
          playerNumber: 1,
          playerType: 'ai',
          aiType: 'heuristic',
          aiDifficulty: 5,
          aiProfileId: null,
          finalEliminatedRings: 10,
          finalTerritorySpaces: 20,
        },
        {
          playerNumber: 2,
          playerType: 'ai',
          aiType: 'heuristic',
          aiDifficulty: 5,
          aiProfileId: null,
          finalEliminatedRings: 5,
          finalTerritorySpaces: 10,
        },
      ],
    };

    // Arrange database mocks for create/update/move + AI user.
    mockGameCreate.mockResolvedValue({ id: 'imported-game-1' } as any);

    const updateCalls: Array<{ data: unknown }> = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    mockGameUpdate.mockImplementation(async (args: any) => {
      updateCalls.push(args);
      return {} as any;
    });

    mockMoveCreate.mockResolvedValue({} as any);
    mockUserFindUnique.mockResolvedValueOnce(null);
    mockUserCreate.mockResolvedValueOnce({ id: 'ai-user-1' } as any);

    const stubService = {
      getGame: jest.fn().mockReturnValue(detail),
      getStateAtMove: jest.fn().mockReturnValue({} as any),
    };

    const gameId = await SelfPlayService.importSelfPlayGameAsGameRecord(
      {
        dbPath: '/tmp/selfplay/games.db',
        gameId: 'sp-1',
      },
      stubService as any
    );

    expect(gameId).toBe('imported-game-1');

    // One of the Game.update calls should be the GameRecordRepository.saveGameRecord
    // write, which includes recordMetadata with source/tags for self-play.
    const recordUpdate = updateCalls.find(
      (call) => (call.data as any).recordMetadata !== undefined
    );
    expect(recordUpdate).toBeDefined();

    const metadata = (recordUpdate!.data as any).recordMetadata as {
      source: string;
      tags: string[];
    };

    expect(metadata.source).toBe('self_play');
    expect(metadata.tags).toEqual(
      expect.arrayContaining([
        'self_play',
        'import:selfplay_sqlite',
        'db:games.db',
        'selfplay_game_id:sp-1',
        'selfplay_source:self_play_pipeline',
        'players:2',
        'winner_seat:1',
        'termination:ring_elimination',
      ])
    );
  });
});
