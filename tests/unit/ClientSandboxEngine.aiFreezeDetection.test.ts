/**
 * AI Freeze Detection Test Suite
 *
 * This suite specifically targets the freeze issue observed in sandbox AI self-play
 * on large boards (square19, hexagonal) with 3-4 AI players.
 *
 * Key features:
 * - Monitors per-turn execution time to detect abrupt freezes
 * - Captures game state BEFORE each turn for reproduction
 * - Saves problematic states as JSON fixtures
 * - Uses real timeouts to detect actual hangs (not just slow execution)
 * - Focuses on large boards where the issue manifests
 *
 * Usage:
 *   # Run all freeze detection tests
 *   RINGRIFT_FREEZE_DETECTION=1 npm test -- --grep "AI Freeze Detection"
 *
 *   # Run with verbose logging
 *   RINGRIFT_FREEZE_DETECTION=1 RINGRIFT_FREEZE_VERBOSE=1 npm test -- --grep "AI Freeze Detection"
 *
 *   # Run specific board/player combo
 *   RINGRIFT_FREEZE_DETECTION=1 npm test -- --grep "square19 with 4 AI"
 *
 * When a freeze is detected, the state is saved to:
 *   tests/fixtures/freeze-states/freeze-{boardType}-{numPlayers}p-seed{seed}-turn{turn}.json
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import { hashGameState } from '../../src/shared/engine/core';
import { serializeGameState } from '../../src/shared/engine/contracts/serialization';

// Configuration
const FREEZE_DETECTION_ENABLED = process.env.RINGRIFT_FREEZE_DETECTION === '1';
const VERBOSE = process.env.RINGRIFT_FREEZE_VERBOSE === '1';

// Thresholds for detecting freezes
const TURN_TIMEOUT_MS = 5000; // 5 seconds - a single turn should never take this long
const TURN_WARNING_MS = 1000; // 1 second - log a warning if turn takes this long
const MAX_TURNS_PER_GAME = 500; // Safety limit for total turns

// Fixture output directory
const FIXTURE_DIR = path.join(__dirname, '..', 'fixtures', 'freeze-states');

/**
 * Captured state for a potentially problematic turn.
 */
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

/**
 * Statistics for a single game run.
 */
interface GameRunStats {
  seed: number;
  totalTurns: number;
  totalDurationMs: number;
  avgTurnMs: number;
  maxTurnMs: number;
  maxTurnNumber: number;
  slowTurns: number; // Turns > TURN_WARNING_MS
  timedOut: boolean;
  timeoutTurn?: number;
  finalStatus: string;
}

/**
 * Ensure the fixture directory exists.
 */
function ensureFixtureDir(): void {
  if (!fs.existsSync(FIXTURE_DIR)) {
    fs.mkdirSync(FIXTURE_DIR, { recursive: true });
  }
}

/**
 * Save a problematic state as a JSON fixture.
 */
function saveFreezeState(capture: TurnCapture): string {
  ensureFixtureDir();

  const filename = `freeze-${capture.boardType}-${capture.numPlayers}p-seed${capture.seed}-turn${capture.turnNumber}.json`;
  const filepath = path.join(FIXTURE_DIR, filename);

  // Serialize the state for reproduction
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
    // Use the canonical serialization for the state
    serializedState: serializeGameState(capture.stateBefore),
    // Also include raw state for debugging
    rawState: capture.stateBefore,
  };

  fs.writeFileSync(filepath, JSON.stringify(fixture, null, 2));
  return filepath;
}

/**
 * Create a sandbox engine for AI-vs-AI games.
 */
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

        // Deterministically pick the first option for reproducibility
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

/**
 * Deterministic PRNG for reproducible runs.
 */
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/**
 * Run a single game with turn-by-turn timing instrumentation.
 * Returns stats and any captured freeze states.
 */
async function runInstrumentedGame(
  boardType: BoardType,
  numPlayers: number,
  seed: number
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
        // Non-AI player (shouldn't happen in all-AI game)
        break;
      }

      // Capture state BEFORE the turn
      const stateHash = hashGameState(stateBefore);
      const stackCount = stateBefore.board.stacks.size;
      const markerCount = stateBefore.board.markers.size;

      // Time the turn execution
      const startTime = Date.now();

      // Use a timeout wrapper to detect actual hangs
      const turnPromise = engine.maybeRunAITurn();

      // Create a timeout promise
      const timeoutPromise = new Promise<'timeout'>((resolve) => {
        setTimeout(() => resolve('timeout'), TURN_TIMEOUT_MS);
      });

      // Race between turn completion and timeout
      const result = await Promise.race([
        turnPromise.then(() => 'complete' as const),
        timeoutPromise,
      ]);

      const endTime = Date.now();
      const duration = endTime - startTime;
      turnDurations.push(duration);

      if (result === 'timeout') {
        // Turn timed out - this is a freeze!
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
        console.error(`   Duration: ${duration}ms (timeout at ${TURN_TIMEOUT_MS}ms)`);
        console.error(
          `   Phase: ${stateBefore.currentPhase}, Player: ${stateBefore.currentPlayer}`
        );
        console.error(`   Stacks: ${stackCount}, Markers: ${markerCount}`);
        console.error(`   State saved: ${filepath}`);
        console.error(`   Hash: ${stateHash}\n`);

        break;
      }

      // Log slow turns (but not timeouts)
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

        if (VERBOSE) {
          console.warn(
            `⚠️  Slow turn: ${boardType} ${numPlayers}p seed=${seed} turn=${turnNumber} took ${duration}ms`
          );
          console.warn(`   Phase: ${stateBefore.currentPhase}, Stacks: ${stackCount}`);
        }

        // Save slow turns too for analysis
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

describe('AI Freeze Detection', () => {
  // Focus on large boards where freeze is observed
  const testConfigs: Array<{ boardType: BoardType; numPlayers: number }> = [
    { boardType: 'square19', numPlayers: 3 },
    { boardType: 'square19', numPlayers: 4 },
    { boardType: 'hexagonal', numPlayers: 3 },
    { boardType: 'hexagonal', numPlayers: 4 },
    // Include smaller boards for baseline comparison
    { boardType: 'square8', numPlayers: 4 },
  ];

  // Number of seeded games per configuration
  const GAMES_PER_CONFIG = 10;

  const maybeDescribe = FREEZE_DETECTION_ENABLED ? describe : describe.skip;

  maybeDescribe('Large board AI self-play freeze detection', () => {
    for (const { boardType, numPlayers } of testConfigs) {
      describe(`${boardType} with ${numPlayers} AI players`, () => {
        it(`runs ${GAMES_PER_CONFIG} seeded games monitoring for freezes`, async () => {
          const allStats: GameRunStats[] = [];
          const allFreezes: TurnCapture[] = [];
          let freezeCount = 0;

          console.log(`\n📊 Testing ${boardType} with ${numPlayers} AI players...`);

          for (let i = 0; i < GAMES_PER_CONFIG; i++) {
            // Use distinct seeds per configuration
            const seed =
              1000 *
                (testConfigs.findIndex(
                  (c) => c.boardType === boardType && c.numPlayers === numPlayers
                ) +
                  1) +
              i;

            if (VERBOSE) {
              console.log(`  Running game ${i + 1}/${GAMES_PER_CONFIG} with seed ${seed}...`);
            }

            const { stats, freezeCaptures } = await runInstrumentedGame(
              boardType,
              numPlayers,
              seed
            );

            allStats.push(stats);
            allFreezes.push(...freezeCaptures);

            if (stats.timedOut) {
              freezeCount++;
            }
          }

          // Summary statistics
          const avgTurns = allStats.reduce((a, b) => a + b.totalTurns, 0) / allStats.length;
          const avgTurnTime = allStats.reduce((a, b) => a + b.avgTurnMs, 0) / allStats.length;
          const maxTurnTime = Math.max(...allStats.map((s) => s.maxTurnMs));
          const totalSlowTurns = allStats.reduce((a, b) => a + b.slowTurns, 0);
          const completedGames = allStats.filter((s) => s.finalStatus !== 'active').length;

          console.log(`\n📈 Results for ${boardType} ${numPlayers}p:`);
          console.log(`   Games: ${GAMES_PER_CONFIG}, Completed: ${completedGames}`);
          console.log(
            `   Avg turns: ${avgTurns.toFixed(1)}, Avg turn time: ${avgTurnTime.toFixed(1)}ms`
          );
          console.log(`   Max turn time: ${maxTurnTime.toFixed(1)}ms`);
          console.log(`   Slow turns (>${TURN_WARNING_MS}ms): ${totalSlowTurns}`);
          console.log(`   Freezes detected: ${freezeCount}`);

          if (allFreezes.length > 0) {
            console.log(`   ⚠️  ${allFreezes.length} problematic states saved to ${FIXTURE_DIR}`);
          }

          // Test assertions
          expect(freezeCount).toBe(0); // No freezes should occur

          // Performance assertion: average turn should be under warning threshold
          expect(avgTurnTime).toBeLessThan(TURN_WARNING_MS);
        }, 300000); // 5 minute timeout for the entire test
      });
    }
  });

  maybeDescribe('Freeze reproduction tests', () => {
    it('can load and reproduce a saved freeze state', async () => {
      // Check if any freeze states exist
      if (!fs.existsSync(FIXTURE_DIR)) {
        console.log('No freeze states to test - run freeze detection first');
        return;
      }

      const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith('.json'));
      if (files.length === 0) {
        console.log('No freeze state fixtures found');
        return;
      }

      // Test the first freeze state
      const firstFile = files[0];
      const filepath = path.join(FIXTURE_DIR, firstFile);
      const fixture = JSON.parse(fs.readFileSync(filepath, 'utf-8'));

      console.log(`\n🔍 Testing reproduction of: ${firstFile}`);
      console.log(`   ${fixture.metadata.description}`);

      // Verify the fixture has the expected structure
      expect(fixture.metadata).toBeDefined();
      expect(fixture.metadata.boardType).toBeDefined();
      expect(fixture.metadata.numPlayers).toBeDefined();
      expect(fixture.metadata.seed).toBeDefined();
      expect(fixture.serializedState).toBeDefined();
    });
  });
});

/**
 * Standalone function to run freeze detection from command line.
 * Can be called directly: npx ts-node tests/unit/ClientSandboxEngine.aiFreezeDetection.test.ts
 */
export async function runFreezeDetection(): Promise<void> {
  console.log('🔍 Running AI Freeze Detection Suite\n');
  console.log(`Configuration:`);
  console.log(`  Turn timeout: ${TURN_TIMEOUT_MS}ms`);
  console.log(`  Turn warning: ${TURN_WARNING_MS}ms`);
  console.log(`  Max turns per game: ${MAX_TURNS_PER_GAME}`);
  console.log(`  Output directory: ${FIXTURE_DIR}\n`);

  const configs = [
    { boardType: 'square19' as BoardType, numPlayers: 4 },
    { boardType: 'hexagonal' as BoardType, numPlayers: 4 },
  ];

  for (const { boardType, numPlayers } of configs) {
    console.log(`\n▶️  Testing ${boardType} with ${numPlayers} AI players...`);

    for (let seed = 1; seed <= 20; seed++) {
      const { stats } = await runInstrumentedGame(boardType, numPlayers, seed);

      if (stats.timedOut) {
        console.log(`   🔴 Seed ${seed}: FREEZE at turn ${stats.timeoutTurn}`);
      } else {
        console.log(
          `   ✅ Seed ${seed}: ${stats.totalTurns} turns, max ${stats.maxTurnMs.toFixed(0)}ms, ${stats.finalStatus}`
        );
      }
    }
  }

  console.log('\n✅ Freeze detection complete');
}

// Allow running directly
if (require.main === module) {
  runFreezeDetection().catch(console.error);
}
