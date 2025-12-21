#!/usr/bin/env ts-node
/**
 * Generate Golden Game Fixtures
 *
 * Creates new game fixtures by running games through the FSM-enabled TypeScript
 * engine with AI-controlled moves. These fixtures are guaranteed to be compatible
 * with FSM orchestration since they're generated using it.
 *
 * Usage:
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/generate-golden-fixtures.ts
 *
 * Options:
 *   --limit <n>           Games per board type (default: 2)
 *   --output <dir>        Output directory (default: tests/fixtures/golden-games)
 *   --boards <list>       Comma-separated board types (default: square8)
 *   --max-turns <n>       Maximum turns per game (default: 100)
 */

import * as fs from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import type { BoardType, GameState, Move, Player, TimeControl } from '../src/shared/types/game';
import type {
  GameRecord,
  MoveRecord,
  PlayerRecordInfo,
  GameOutcome,
} from '../src/shared/types/gameRecord';
import { createInitialGameState } from '../src/shared/engine/initialState';
import { processTurn, getValidMoves } from '../src/shared/engine/orchestration/turnOrchestrator';
import { moveToMoveRecord } from '../src/shared/types/gameRecord';

interface CliArgs {
  limit: number;
  outputDir: string;
  boardTypes: BoardType[];
  maxTurns: number;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    limit: 2,
    outputDir: 'tests/fixtures/golden-games',
    boardTypes: ['square8'],
    maxTurns: 100,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--limit' && argv[i + 1]) {
      args.limit = parseInt(argv[i + 1], 10);
      i++;
    } else if (arg === '--output' && argv[i + 1]) {
      args.outputDir = argv[i + 1];
      i++;
    } else if (arg === '--boards' && argv[i + 1]) {
      args.boardTypes = argv[i + 1].split(',') as BoardType[];
      i++;
    } else if (arg === '--max-turns' && argv[i + 1]) {
      args.maxTurns = parseInt(argv[i + 1], 10);
      i++;
    }
  }

  return args;
}

/**
 * Create players for a game
 */
function createPlayers(numPlayers: number): Player[] {
  return Array.from({ length: numPlayers }, (_, i) => ({
    id: `player-${i + 1}`,
    username: `Player ${i + 1}`,
    type: 'ai' as const,
    playerNumber: i + 1,
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
  }));
}

/**
 * Select a move using simple random selection from valid moves.
 * Prioritizes actual actions over no-op moves when available.
 */
function selectMove(validMoves: Move[]): Move | null {
  if (validMoves.length === 0) return null;

  // Prefer action moves over no-op moves
  const actionMoves = validMoves.filter(
    (m) =>
      m.type !== 'no_placement_action' &&
      m.type !== 'no_movement_action' &&
      m.type !== 'no_line_action' &&
      m.type !== 'no_territory_action'
  );

  const candidates = actionMoves.length > 0 ? actionMoves : validMoves;

  // Random selection
  const index = Math.floor(Math.random() * candidates.length);
  return candidates[index];
}

/**
 * Generate a no-action bookkeeping move for phases where getValidMoves returns empty.
 * Per RR-CANON-R076, the core layer doesn't fabricate these - hosts must construct them.
 */
function generateNoActionMove(state: GameState): Move | null {
  const player = state.currentPlayer;
  const moveNumber = state.moveHistory.length + 1;
  const phase = state.currentPhase;

  switch (phase) {
    case 'ring_placement':
      return {
        id: `no-placement-${moveNumber}`,
        type: 'no_placement_action',
        player,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;
    case 'movement':
      return {
        id: `no-movement-${moveNumber}`,
        type: 'no_movement_action',
        player,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;
    case 'capture':
      // No captures available, skip to line processing
      return {
        id: `skip-capture-${moveNumber}`,
        type: 'skip_capture',
        player,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;
    case 'line_processing':
      return {
        id: `no-line-${moveNumber}`,
        type: 'no_line_action',
        player,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;
    case 'territory_processing':
      return {
        id: `no-territory-${moveNumber}`,
        type: 'no_territory_action',
        player,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;
    default:
      return null;
  }
}

/**
 * Run a game to completion and return the game record
 */
function runGame(
  gameId: string,
  boardType: BoardType,
  numPlayers: number,
  maxTurns: number
): GameRecord | null {
  const timeControl: TimeControl = { type: 'rapid', initialTime: 600, increment: 0 };
  const players = createPlayers(numPlayers);

  let state = createInitialGameState(gameId, boardType, players, timeControl);
  const moves: MoveRecord[] = [];
  const startTime = new Date();

  let turnCount = 0;
  let moveNumber = 1;

  while (state.currentPhase !== 'game_over' && turnCount < maxTurns) {
    // Check if game has ended
    if (state.gameStatus === 'completed' || state.gameStatus === 'finished') {
      break;
    }
    const validMoves = getValidMoves(state);
    let selectedMove: Move | null = null;

    if (validMoves.length === 0) {
      // No interactive moves - try to generate a no-action bookkeeping move
      selectedMove = generateNoActionMove(state);
      if (!selectedMove) {
        console.log(`[generate] No valid moves at turn ${turnCount}, phase ${state.currentPhase}`);
        break;
      }
    } else {
      selectedMove = selectMove(validMoves);
      if (!selectedMove) {
        console.log(`[generate] Failed to select move at turn ${turnCount}`);
        break;
      }
    }

    // Add move number and timestamp
    const move: Move = {
      ...selectedMove,
      moveNumber,
      timestamp: new Date(),
      thinkTime: 0,
    };

    try {
      const result = processTurn(state, move);
      state = result.nextState;

      // Record the move
      moves.push(moveToMoveRecord(move));
      moveNumber++;

      // Count turns based on player transitions
      if (result.nextState.currentPhase === 'ring_placement') {
        turnCount++;
      }
    } catch (error) {
      console.error(`[generate] Error processing move at turn ${turnCount}:`, error);
      break;
    }
  }

  if (moves.length < 10) {
    console.log(`[generate] Game too short (${moves.length} moves), skipping`);
    return null;
  }

  const endTime = new Date();

  // Build player records
  const playerRecords: PlayerRecordInfo[] = state.players.map((p) => ({
    playerNumber: p.playerNumber,
    username: `Player ${p.playerNumber}`,
    playerType: 'ai' as const,
    aiType: 'random',
    aiDifficulty: 1,
  }));

  // Determine outcome
  let outcome: GameOutcome = 'ring_elimination';
  if (state.currentPhase === 'game_over' || state.gameStatus === 'completed') {
    // Check what caused the game to end
    const hasEliminated = state.players.some((p) => p.eliminatedRings > 0);
    const hasTerritory = state.players.some((p) => p.territorySpaces > 0);
    if (hasTerritory && !hasEliminated) {
      outcome = 'territory_control';
    }
  }

  // Build final score
  const finalScore = {
    ringsEliminated: {} as Record<number, number>,
    territorySpaces: {} as Record<number, number>,
    ringsRemaining: {} as Record<number, number>,
  };

  for (const p of state.players) {
    finalScore.ringsEliminated[p.playerNumber] = p.eliminatedRings;
    finalScore.territorySpaces[p.playerNumber] = p.territorySpaces;
    finalScore.ringsRemaining[p.playerNumber] = p.ringsInHand;
  }

  const gameRecord: GameRecord = {
    id: gameId,
    boardType,
    numPlayers,
    isRated: false,
    players: playerRecords,
    ...(state.winner !== undefined && { winner: state.winner }),
    outcome,
    finalScore,
    startedAt: startTime,
    endedAt: endTime,
    totalMoves: moves.length,
    totalDurationMs: endTime.getTime() - startTime.getTime(),
    moves,
    metadata: {
      recordVersion: '1.0.0',
      createdAt: new Date().toISOString(),
      source: 'self_play',
      tags: ['golden', 'fsm-canonical', 'generated'],
    },
  };

  return gameRecord;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  console.log('[generate] Golden fixture generation starting...');
  console.log(`[generate] Output directory: ${args.outputDir}`);
  console.log(`[generate] Games per board type: ${args.limit}`);
  console.log(`[generate] Board types: ${args.boardTypes.join(', ')}`);
  console.log(`[generate] Max turns per game: ${args.maxTurns}`);

  // Create output directory
  const outputDir = path.resolve(args.outputDir);
  fs.mkdirSync(outputDir, { recursive: true });

  let generatedCount = 0;
  let failedCount = 0;

  for (const boardType of args.boardTypes) {
    for (let numPlayers = 2; numPlayers <= 2; numPlayers++) {
      // Start with 2 players
      for (let i = 0; i < args.limit; i++) {
        const gameId = uuidv4();
        console.log(
          `[generate] Running game ${i + 1}/${args.limit} for ${boardType} ${numPlayers}p...`
        );

        try {
          const gameRecord = runGame(gameId, boardType, numPlayers, args.maxTurns);

          if (gameRecord) {
            // Generate filename
            const hasCaptures = gameRecord.moves.some(
              (m) => m.type === 'overtaking_capture' || m.type === 'continue_capture_segment'
            );
            const hasLines = gameRecord.moves.some(
              (m) => m.type === 'process_line' || m.type === 'choose_line_option'
            );
            const hasTerritory = gameRecord.moves.some(
              (m) => m.type === 'choose_territory_option' || m.type === 'eliminate_rings_from_stack'
            );

            let category = 'full_game';
            if (hasTerritory) category = 'territory';
            else if (hasLines) category = 'line_formation';
            else if (hasCaptures) category = 'captures';

            const filename = `${category}_${boardType}_${numPlayers}p_${gameId.slice(0, 8)}.json`;
            const outputPath = path.join(outputDir, filename);

            fs.writeFileSync(outputPath, JSON.stringify(gameRecord, null, 2));
            console.log(`[generate] Wrote ${filename} (${gameRecord.totalMoves} moves)`);
            generatedCount++;
          } else {
            failedCount++;
          }
        } catch (error) {
          console.error(`[generate] Error generating game:`, error);
          failedCount++;
        }
      }
    }
  }

  console.log(
    `\n[generate] Generation complete: ${generatedCount} games generated, ${failedCount} failed`
  );
}

main().catch((error) => {
  console.error('[generate] Fatal error:', error);
  process.exitCode = 1;
});
