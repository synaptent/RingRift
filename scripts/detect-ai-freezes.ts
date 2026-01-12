#!/usr/bin/env npx ts-node
/**
 * AI Freeze Detection Script
 *
 * Runs AI-vs-AI games on large boards to detect and capture freeze conditions.
 *
 * Usage:
 *   npx ts-node scripts/detect-ai-freezes.ts [options]
 *
 * Options:
 *   --board <type>     Board type: square19, hexagonal, square8 (default: square19)
 *   --players <n>      Number of players: 2, 3, 4 (default: 4)
 *   --games <n>        Number of games to run (default: 50)
 *   --seed <n>         Starting seed (default: 1)
 *   --timeout <ms>     Turn timeout in milliseconds (default: 5000)
 *   --verbose          Enable verbose logging
 *
 * Examples:
 *   npx ts-node scripts/detect-ai-freezes.ts --board square19 --players 4 --games 100
 *   npx ts-node scripts/detect-ai-freezes.ts --board hexagonal --players 3 --verbose
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../src/shared/types/game';
import { hashGameState } from '../src/shared/engine/core';
import { serializeGameState } from '../src/shared/engine/contracts/serialization';

// Parse command line arguments
function parseArgs(): {
  boardType: BoardType;
  numPlayers: number;
  numGames: number;
  startingSeed: number;
  turnTimeoutMs: number;
  verbose: boolean;
} {
  const args = process.argv.slice(2);
  const config = {
    boardType: 'square19' as BoardType,
    numPlayers: 4,
    numGames: 50,
    startingSeed: 1,
    turnTimeoutMs: 5000,
    verbose: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const nextArg = args[i + 1];

    switch (arg) {
      case '--board':
        if (['square8', 'square19', 'hexagonal', 'hex8'].includes(nextArg)) {
          config.boardType = nextArg as BoardType;
        }
        i++;
        break;
      case '--players':
        const players = parseInt(nextArg, 10);
        if (players >= 2 && players <= 4) {
          config.numPlayers = players;
        }
        i++;
        break;
      case '--games':
        config.numGames = parseInt(nextArg, 10) || 50;
        i++;
        break;
      case '--seed':
        config.startingSeed = parseInt(nextArg, 10) || 1;
        i++;
        break;
      case '--timeout':
        config.turnTimeoutMs = parseInt(nextArg, 10) || 5000;
        i++;
        break;
      case '--verbose':
        config.verbose = true;
        break;
      case '--help':
        console.log(`
AI Freeze Detection Script

Usage:
  npx ts-node scripts/detect-ai-freezes.ts [options]

Options:
  --board <type>     Board type: square19, hexagonal, square8 (default: square19)
  --players <n>      Number of players: 2, 3, 4 (default: 4)
  --games <n>        Number of games to run (default: 50)
  --seed <n>         Starting seed (default: 1)
  --timeout <ms>     Turn timeout in milliseconds (default: 5000)
  --verbose          Enable verbose logging
        `);
        process.exit(0);
    }
  }

  return config;
}

// Configuration
const TURN_WARNING_MS = 1000;
const MAX_TURNS_PER_GAME = 500;

// Fixture output directory
const FIXTURE_DIR = path.join(__dirname, '..', 'tests', 'fixtures', 'freeze-states');

interface TurnCapture {
  boardType: BoardType;
  numPlayers: number;
  seed: number;
  turnNumber: number;
  durationMs: number;
  stateBefore: GameState;
  stateHash: string;
  currentPhase: string;
  currentPlayer: number;
  stackCount: number;
  markerCount: number;
  timestamp: string;
}

interface GameRunStats {
  seed: number;
  totalTurns: number;
  totalDurationMs: number;
  avgTurnMs: number;
  maxTurnMs: number;
  maxTurnNumber: number;
  slowTurns: number;
  timedOut: boolean;
  timeoutTurn?: number;
  finalStatus: string;
}

function ensureFixtureDir(): void {
  if (!fs.existsSync(FIXTURE_DIR)) {
    fs.mkdirSync(FIXTURE_DIR, { recursive: true });
  }
}

function saveFreezeState(capture: TurnCapture): string {
  ensureFixtureDir();

  const filename = `freeze-${capture.boardType}-${capture.numPlayers}p-seed${capture.seed}-turn${capture.turnNumber}.json`;
  const filepath = path.join(FIXTURE_DIR, filename);

  const fixture = {
    metadata: {
      boardType: capture.boardType,
      numPlayers: capture.numPlayers,
      seed: capture.seed,
      turnNumber: capture.turnNumber,
      durationMs: capture.durationMs,
      currentPhase: capture.currentPhase,
      currentPlayer: capture.currentPlayer,
      stackCount: capture.stackCount,
      markerCount: capture.markerCount,
      stateHash: capture.stateHash,
      capturedAt: capture.timestamp,
      description: `Freeze detected at turn ${capture.turnNumber} (${capture.durationMs}ms) on ${capture.boardType} with ${capture.numPlayers} players`,
    },
    serializedState: serializeGameState(capture.stateBefore),
    rawState: capture.stateBefore,
  };

  fs.writeFileSync(filepath, JSON.stringify(fixture, null, 2));
  return filepath;
}

function createEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers,
    playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
  };

  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice extends PlayerChoice>(
      choice: TChoice
    ): Promise<PlayerChoiceResponseFor<TChoice>> {
      const anyChoice = choice as any;

      if (anyChoice.type === 'capture_direction') {
        const cd = anyChoice as CaptureDirectionChoice;
        const options = cd.options || [];
        if (options.length === 0) {
          throw new Error('No options for capture_direction');
        }

        return {
          choiceId: cd.id,
          playerNumber: cd.playerNumber,
          choiceType: cd.type,
          selectedOption: options[0],
        } as PlayerChoiceResponseFor<TChoice>;
      }

      const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption,
      } as PlayerChoiceResponseFor<TChoice>;
    },
  };

  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

async function runInstrumentedGame(
  boardType: BoardType,
  numPlayers: number,
  seed: number,
  turnTimeoutMs: number,
  verbose: boolean
): Promise<{ stats: GameRunStats; freezeCaptures: TurnCapture[] }> {
  const rng = makePrng(seed);
  const originalRandom = Math.random;
  Math.random = rng;

  const freezeCaptures: TurnCapture[] = [];
  const turnDurations: number[] = [];

  let timedOut = false;
  let timeoutTurn: number | undefined;

  try {
    const engine = createEngine(boardType, numPlayers);
    let turnNumber = 0;

    while (turnNumber < MAX_TURNS_PER_GAME) {
      const stateBefore = engine.getGameState();

      if (stateBefore.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = stateBefore.players.find(
        (p) => p.playerNumber === stateBefore.currentPlayer
      );
      if (!currentPlayer || currentPlayer.type !== 'ai') {
        break;
      }

      const stateHash = hashGameState(stateBefore);
      const stackCount = stateBefore.board.stacks.size;
      const markerCount = stateBefore.board.markers.size;

      const startTime = Date.now();

      // Use a timeout wrapper to detect actual hangs
      const turnPromise = engine.maybeRunAITurn();
      const timeoutPromise = new Promise<'timeout'>((resolve) => {
        setTimeout(() => resolve('timeout'), turnTimeoutMs);
      });

      const result = await Promise.race([
        turnPromise.then(() => 'complete' as const),
        timeoutPromise,
      ]);

      const endTime = Date.now();
      const duration = endTime - startTime;
      turnDurations.push(duration);

      if (result === 'timeout') {
        timedOut = true;
        timeoutTurn = turnNumber;

        const capture: TurnCapture = {
          boardType,
          numPlayers,
          seed,
          turnNumber,
          durationMs: duration,
          stateBefore,
          stateHash,
          currentPhase: stateBefore.currentPhase,
          currentPlayer: stateBefore.currentPlayer,
          stackCount,
          markerCount,
          timestamp: new Date().toISOString(),
        };

        freezeCaptures.push(capture);
        const filepath = saveFreezeState(capture);

        console.error(
          `\n🔴 FREEZE DETECTED: ${boardType} ${numPlayers}p seed=${seed} turn=${turnNumber}`
        );
        console.error(`   Duration: ${duration}ms (timeout at ${turnTimeoutMs}ms)`);
        console.error(
          `   Phase: ${stateBefore.currentPhase}, Player: ${stateBefore.currentPlayer}`
        );
        console.error(`   Stacks: ${stackCount}, Markers: ${markerCount}`);
        console.error(`   State saved: ${filepath}`);
        console.error(`   Hash: ${stateHash}\n`);

        break;
      }

      if (duration >= TURN_WARNING_MS) {
        const capture: TurnCapture = {
          boardType,
          numPlayers,
          seed,
          turnNumber,
          durationMs: duration,
          stateBefore,
          stateHash,
          currentPhase: stateBefore.currentPhase,
          currentPlayer: stateBefore.currentPlayer,
          stackCount,
          markerCount,
          timestamp: new Date().toISOString(),
        };
        freezeCaptures.push(capture);

        if (verbose) {
          console.warn(
            `⚠️  Slow turn: seed=${seed} turn=${turnNumber} took ${duration}ms (phase: ${stateBefore.currentPhase}, stacks: ${stackCount})`
          );
        }

        saveFreezeState(capture);
      }

      turnNumber++;
    }

    const finalState = engine.getGameState();
    const totalDuration = turnDurations.reduce((a, b) => a + b, 0);
    const maxTurnMs = Math.max(...turnDurations, 0);
    const maxTurnIndex = turnDurations.indexOf(maxTurnMs);

    const stats: GameRunStats = {
      seed,
      totalTurns: turnNumber,
      totalDurationMs: totalDuration,
      avgTurnMs: turnNumber > 0 ? totalDuration / turnNumber : 0,
      maxTurnMs,
      maxTurnNumber: maxTurnIndex,
      slowTurns: turnDurations.filter((d) => d >= TURN_WARNING_MS).length,
      timedOut,
      finalStatus: finalState.gameStatus,
    };

    // Only add timeoutTurn if it's defined (exactOptionalPropertyTypes)
    if (timeoutTurn !== undefined) {
      stats.timeoutTurn = timeoutTurn;
    }

    return {
      stats,
      freezeCaptures,
    };
  } finally {
    Math.random = originalRandom;
  }
}

async function main(): Promise<void> {
  const config = parseArgs();

  console.log('🔍 AI Freeze Detection Script\n');
  console.log('Configuration:');
  console.log(`  Board type: ${config.boardType}`);
  console.log(`  Players: ${config.numPlayers}`);
  console.log(`  Games to run: ${config.numGames}`);
  console.log(`  Starting seed: ${config.startingSeed}`);
  console.log(`  Turn timeout: ${config.turnTimeoutMs}ms`);
  console.log(`  Output directory: ${FIXTURE_DIR}`);
  console.log('');

  const allStats: GameRunStats[] = [];
  let freezeCount = 0;
  let slowTurnCount = 0;

  const startTime = Date.now();

  for (let i = 0; i < config.numGames; i++) {
    const seed = config.startingSeed + i;

    process.stdout.write(`\rGame ${i + 1}/${config.numGames} (seed ${seed})...`);

    const { stats } = await runInstrumentedGame(
      config.boardType,
      config.numPlayers,
      seed,
      config.turnTimeoutMs,
      config.verbose
    );

    allStats.push(stats);

    if (stats.timedOut) {
      freezeCount++;
      console.log(`\n   🔴 FREEZE at turn ${stats.timeoutTurn}`);
    } else if (stats.slowTurns > 0) {
      slowTurnCount += stats.slowTurns;
      if (config.verbose) {
        console.log(
          `\n   ⚠️  ${stats.slowTurns} slow turns, max ${stats.maxTurnMs.toFixed(0)}ms at turn ${stats.maxTurnNumber}`
        );
      }
    }
  }

  const totalTime = Date.now() - startTime;

  // Summary
  console.log('\n\n' + '='.repeat(60));
  console.log('📊 SUMMARY');
  console.log('='.repeat(60));

  const completedGames = allStats.filter((s) => s.finalStatus !== 'active').length;
  const avgTurns = allStats.reduce((a, b) => a + b.totalTurns, 0) / allStats.length;
  const avgTurnTime = allStats.reduce((a, b) => a + b.avgTurnMs, 0) / allStats.length;
  const maxTurnTime = Math.max(...allStats.map((s) => s.maxTurnMs));
  const maxTurnSeed = allStats.find((s) => s.maxTurnMs === maxTurnTime)?.seed;

  console.log(`\nGames run: ${config.numGames}`);
  console.log(
    `Games completed: ${completedGames} (${((completedGames / config.numGames) * 100).toFixed(1)}%)`
  );
  console.log(`Total time: ${(totalTime / 1000).toFixed(1)}s`);
  console.log(`\nPerformance:`);
  console.log(`  Average turns per game: ${avgTurns.toFixed(1)}`);
  console.log(`  Average turn time: ${avgTurnTime.toFixed(1)}ms`);
  console.log(`  Maximum turn time: ${maxTurnTime.toFixed(0)}ms (seed ${maxTurnSeed})`);
  console.log(`\nIssues:`);
  console.log(`  Freezes (>${config.turnTimeoutMs}ms): ${freezeCount}`);
  console.log(`  Slow turns (>${TURN_WARNING_MS}ms): ${slowTurnCount}`);

  if (freezeCount > 0) {
    console.log(`\n🔴 ${freezeCount} FREEZE(S) DETECTED!`);
    console.log(`   Check ${FIXTURE_DIR} for captured states`);
    process.exit(1);
  } else if (slowTurnCount > 0) {
    console.log(`\n⚠️  ${slowTurnCount} slow turns detected`);
    console.log(`   Check ${FIXTURE_DIR} for captured states`);
  } else {
    console.log(`\n✅ No freezes detected`);
  }
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
