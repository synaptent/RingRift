#!/usr/bin/env ts-node
/**
 * Curate Golden Game Fixtures
 *
 * Extracts completed games from selfplay databases, replays them through the
 * FSM-enabled TypeScript engine, and exports them as GameRecord JSON files
 * for use in golden replay tests.
 *
 * Usage:
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/curate-golden-fixtures.ts
 *
 * Options:
 *   --db <path>           Path to a specific selfplay database
 *   --limit <n>           Maximum games to curate per category (default: 3)
 *   --output <dir>        Output directory (default: tests/fixtures/golden-games)
 *   --dry-run             List games that would be curated without writing files
 *   --categories <list>   Comma-separated list of categories to curate
 */

import * as fs from 'fs';
import * as path from 'path';
import { getSelfPlayGameService } from '../src/server/services/SelfPlayGameService';
import type { SelfPlayGameDetail } from '../src/server/services/SelfPlayGameService';
import type { BoardType, Move, GameState } from '../src/shared/types/game';
import type {
  GameRecord,
  MoveRecord,
  PlayerRecordInfo,
  GameOutcome,
} from '../src/shared/types/gameRecord';
import { CanonicalReplayEngine } from '../src/shared/replay/CanonicalReplayEngine';
import { buildCanonicalMoveFromSelfPlayRecord } from './selfplay-db-ts-replay';

// Categories for golden games - each should have at least one representative game
type GoldenCategory =
  | 'basic_placement'
  | 'movement'
  | 'captures'
  | 'chain_captures'
  | 'line_formation'
  | 'territory'
  | 'elimination'
  | 'full_game';

interface CuratedGame {
  category: GoldenCategory;
  dbPath: string;
  gameId: string;
  boardType: BoardType;
  numPlayers: number;
  totalMoves: number;
  hasCaptures: boolean;
  hasLines: boolean;
  hasTerritory: boolean;
  winner: number | null;
}

interface CliArgs {
  dbPath?: string;
  limit: number;
  outputDir: string;
  dryRun: boolean;
  categories: GoldenCategory[];
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    limit: 3,
    outputDir: 'tests/fixtures/golden-games',
    dryRun: false,
    categories: [
      'basic_placement',
      'movement',
      'captures',
      'chain_captures',
      'line_formation',
      'territory',
      'elimination',
      'full_game',
    ],
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--db' && argv[i + 1]) {
      args.dbPath = argv[i + 1];
      i++;
    } else if (arg === '--limit' && argv[i + 1]) {
      args.limit = parseInt(argv[i + 1], 10);
      i++;
    } else if (arg === '--output' && argv[i + 1]) {
      args.outputDir = argv[i + 1];
      i++;
    } else if (arg === '--dry-run') {
      args.dryRun = true;
    } else if (arg === '--categories' && argv[i + 1]) {
      args.categories = argv[i + 1].split(',') as GoldenCategory[];
      i++;
    }
  }

  return args;
}

/**
 * Categorize a game based on its move patterns
 */
function categorizeGame(detail: SelfPlayGameDetail): GoldenCategory[] {
  const categories: GoldenCategory[] = [];
  const moveTypes = detail.moves.map((m) => m.moveType);

  // Check for specific move patterns
  const hasPlacement = moveTypes.some((t) => t === 'place_ring');
  const hasMovement = moveTypes.some((t) => t === 'move_stack');
  const hasCapture = moveTypes.some(
    (t) => t === 'overtaking_capture' || t === 'continue_capture_segment'
  );
  const hasChainCapture = moveTypes.some((t) => t === 'continue_capture_segment');
  const hasLine = moveTypes.some((t) => t === 'process_line' || t === 'choose_line_option');
  const hasTerritory = moveTypes.some(
    (t) => t === 'choose_territory_option' || t === 'eliminate_rings_from_stack'
  );

  // Basic placement: games with only placement and simple movement
  if (hasPlacement && !hasCapture && !hasLine && !hasTerritory && detail.moves.length <= 20) {
    categories.push('basic_placement');
  }

  // Movement: games with movement but no complex mechanics
  if (hasMovement && !hasCapture && !hasLine && !hasTerritory) {
    categories.push('movement');
  }

  // Captures: games with captures
  if (hasCapture && !hasChainCapture) {
    categories.push('captures');
  }

  // Chain captures: games with chain captures
  if (hasChainCapture) {
    categories.push('chain_captures');
  }

  // Line formation: games with line processing
  if (hasLine) {
    categories.push('line_formation');
  }

  // Territory: games with territory processing
  if (hasTerritory) {
    categories.push('territory');
  }

  // Elimination: games that ended with a player having no rings
  const eliminationMoves = moveTypes.filter(
    (t) => t === 'eliminate_rings_from_stack' || t === 'forced_elimination'
  );
  if (eliminationMoves.length > 0) {
    categories.push('elimination');
  }

  // Full game: completed games with winner
  if (detail.winner !== null && detail.completedAt !== null) {
    categories.push('full_game');
  }

  return categories;
}

/**
 * Convert SelfPlayGameDetail to GameRecord through FSM replay
 */
async function convertToGameRecord(
  detail: SelfPlayGameDetail,
  dbPath: string
): Promise<GameRecord | null> {
  try {
    const boardType = detail.boardType as BoardType;
    const numPlayers = detail.numPlayers;

    // Create replay engine with FSM orchestration
    const engine = new CanonicalReplayEngine({
      gameId: detail.gameId,
      boardType,
      numPlayers,
      initialState: detail.initialState,
    });

    // Replay all moves
    const moveRecords: MoveRecord[] = [];

    for (let i = 0; i < detail.moves.length; i++) {
      const selfPlayMove = detail.moves[i];
      const move = buildCanonicalMoveFromSelfPlayRecord(selfPlayMove, i + 1);

      const result = await engine.applyMove(move);
      if (!result.success) {
        console.error(
          `[curate] Failed to replay move ${i + 1} in game ${detail.gameId}: ${result.error}`
        );
        return null;
      }

      // Convert to MoveRecord
      moveRecords.push({
        moveNumber: move.moveNumber,
        player: move.player,
        type: move.type,
        thinkTimeMs: move.thinkTime ?? 0,
        ...(move.from !== undefined && { from: move.from }),
        ...(move.to !== undefined && { to: move.to }),
        ...(move.captureTarget !== undefined && { captureTarget: move.captureTarget }),
        ...(move.formedLines !== undefined && { formedLines: move.formedLines }),
        ...(move.collapsedMarkers !== undefined && { collapsedMarkers: move.collapsedMarkers }),
        ...(move.disconnectedRegions !== undefined && {
          disconnectedRegions: move.disconnectedRegions,
        }),
        ...(move.eliminatedRings !== undefined && { eliminatedRings: move.eliminatedRings }),
      });
    }

    const finalState = engine.getState() as GameState;

    // Build player records
    const players: PlayerRecordInfo[] = detail.players.map((p) => ({
      playerNumber: p.playerNumber,
      username: `Player ${p.playerNumber}`,
      playerType: p.playerType === 'ai' ? 'ai' : 'human',
      ...(p.aiDifficulty !== null && { aiDifficulty: p.aiDifficulty }),
      ...(p.aiType !== null && { aiType: p.aiType }),
    }));

    // Determine outcome
    let outcome: GameOutcome = 'ring_elimination';
    if (detail.terminationReason) {
      const reason = detail.terminationReason.toLowerCase();
      if (reason.includes('territory')) {
        outcome = 'territory_control';
      } else if (reason.includes('last_player') || reason.includes('lps')) {
        outcome = 'last_player_standing';
      } else if (reason.includes('resign')) {
        outcome = 'resignation';
      } else if (reason.includes('timeout')) {
        outcome = 'timeout';
      } else if (reason.includes('draw')) {
        outcome = 'draw';
      }
    }

    // Build final score from player data
    const finalScore = {
      ringsEliminated: {} as Record<number, number>,
      territorySpaces: {} as Record<number, number>,
      ringsRemaining: {} as Record<number, number>,
    };

    for (const p of detail.players) {
      finalScore.ringsEliminated[p.playerNumber] = p.finalEliminatedRings ?? 0;
      finalScore.territorySpaces[p.playerNumber] = p.finalTerritorySpaces ?? 0;
      // Calculate rings remaining from final state
      const playerState = finalState.players.find((fp) => fp.playerNumber === p.playerNumber);
      if (playerState) {
        finalScore.ringsRemaining[p.playerNumber] = playerState.ringsInHand;
      }
    }

    const startedAt = new Date(detail.createdAt);
    const endedAt = detail.completedAt ? new Date(detail.completedAt) : new Date();

    const gameRecord: GameRecord = {
      id: detail.gameId,
      boardType,
      numPlayers,
      isRated: false,
      players,
      // Use conditional spread to avoid undefined assignment
      ...(detail.winner !== null && { winner: detail.winner }),
      outcome,
      finalScore,
      startedAt,
      endedAt,
      totalMoves: moveRecords.length,
      totalDurationMs: detail.durationMs ?? endedAt.getTime() - startedAt.getTime(),
      moves: moveRecords,
      metadata: {
        recordVersion: '1.0.0',
        createdAt: new Date().toISOString(),
        source: 'self_play',
        sourceId: dbPath,
        tags: ['golden', 'fsm-canonical'],
      },
    };

    return gameRecord;
  } catch (error) {
    console.error(`[curate] Error converting game ${detail.gameId}:`, error);
    return null;
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const service = getSelfPlayGameService();

  console.log('[curate] Golden fixture curation starting...');
  console.log(`[curate] Output directory: ${args.outputDir}`);
  console.log(`[curate] Limit per category: ${args.limit}`);
  console.log(`[curate] Categories: ${args.categories.join(', ')}`);
  console.log(`[curate] Dry run: ${args.dryRun}`);

  // Find databases to search
  const dbPaths: string[] = [];
  if (args.dbPath) {
    dbPaths.push(path.resolve(args.dbPath));
  } else {
    // Search for databases in standard locations
    const rootDir = process.cwd();
    const databases = service.listDatabases(rootDir);
    dbPaths.push(...databases.map((db) => db.path));
  }

  console.log(`[curate] Found ${dbPaths.length} database(s) to search`);

  // Track games by category
  const gamesByCategory = new Map<GoldenCategory, CuratedGame[]>();
  for (const cat of args.categories) {
    gamesByCategory.set(cat, []);
  }

  // Search each database for candidate games
  for (const dbPath of dbPaths) {
    console.log(`[curate] Searching ${path.basename(dbPath)}...`);

    try {
      const games = service.listGames(dbPath, { hasWinner: true, limit: 100 });

      for (const gameSummary of games) {
        // Get full game details for categorization
        const detail = service.getGame(dbPath, gameSummary.gameId);
        if (!detail || !detail.moves || detail.moves.length === 0) continue;

        const categories = categorizeGame(detail);

        for (const category of categories) {
          if (!args.categories.includes(category)) continue;

          const categoryGames = gamesByCategory.get(category)!;
          if (categoryGames.length >= args.limit) continue;

          categoryGames.push({
            category,
            dbPath,
            gameId: detail.gameId,
            boardType: detail.boardType as BoardType,
            numPlayers: detail.numPlayers,
            totalMoves: detail.moves.length,
            hasCaptures: detail.moves.some(
              (m) =>
                m.moveType === 'overtaking_capture' || m.moveType === 'continue_capture_segment'
            ),
            hasLines: detail.moves.some(
              (m) => m.moveType === 'process_line' || m.moveType === 'choose_line_option'
            ),
            hasTerritory: detail.moves.some(
              (m) =>
                m.moveType === 'choose_territory_option' ||
                m.moveType === 'eliminate_rings_from_stack'
            ),
            winner: detail.winner,
          });
        }
      }
    } catch (error) {
      console.error(`[curate] Error searching ${dbPath}:`, error);
    }
  }

  // Report what was found
  console.log('\n[curate] Games found by category:');
  let totalFound = 0;
  for (const [category, games] of gamesByCategory) {
    console.log(`  ${category}: ${games.length} game(s)`);
    totalFound += games.length;
  }

  if (totalFound === 0) {
    console.log('[curate] No suitable games found for curation.');
    return;
  }

  if (args.dryRun) {
    console.log('\n[curate] Dry run - would curate the following games:');
    for (const [category, games] of gamesByCategory) {
      for (const game of games) {
        console.log(`  [${category}] ${game.gameId} (${game.boardType}, ${game.numPlayers}p)`);
      }
    }
    return;
  }

  // Create output directory
  const outputDir = path.resolve(args.outputDir);
  fs.mkdirSync(outputDir, { recursive: true });

  // Process and write games
  let curatedCount = 0;
  let failedCount = 0;
  const processedGameIds = new Set<string>();

  for (const [category, games] of gamesByCategory) {
    for (const game of games) {
      // Skip if already processed (game may be in multiple categories)
      if (processedGameIds.has(game.gameId)) continue;
      processedGameIds.add(game.gameId);

      console.log(`[curate] Processing ${game.gameId} (${category})...`);

      const detail = service.getGame(game.dbPath, game.gameId);
      if (!detail) {
        console.error(`[curate] Could not load game ${game.gameId}`);
        failedCount++;
        continue;
      }

      const gameRecord = await convertToGameRecord(detail, game.dbPath);
      if (!gameRecord) {
        failedCount++;
        continue;
      }

      // Generate filename: category_boardtype_numplayers_gameid.json
      const filename = `${category}_${game.boardType}_${game.numPlayers}p_${game.gameId.slice(0, 8)}.json`;
      const outputPath = path.join(outputDir, filename);

      fs.writeFileSync(outputPath, JSON.stringify(gameRecord, null, 2));
      console.log(`[curate] Wrote ${filename}`);
      curatedCount++;
    }
  }

  console.log(`\n[curate] Curation complete: ${curatedCount} games curated, ${failedCount} failed`);

  // Cleanup
  service.closeAll();
}

main().catch((error) => {
  console.error('[curate] Fatal error:', error);
  process.exitCode = 1;
});
