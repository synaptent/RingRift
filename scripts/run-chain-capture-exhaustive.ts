#!/usr/bin/env ts-node
/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Chain Capture Exhaustive Generator
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Offline diagnostic script that generates random board configurations and
 * exhaustively enumerates all possible capture sequences (including chains).
 *
 * This script was recreated based on a removed heavy generator that caused
 * OOM crashes and Jest worker failures in CI. It is designed to run offline
 * as a standalone diagnostic tool.
 *
 * Configuration:
 * - Places one overtaking stack (attacker) with height 2-3
 * - Places one guaranteed capturable target with height 2-3
 * - Places 2-3 additional stacks at random positions with height 2-3
 * - Runs on square8, square19, and hexagonal boards
 * - Exhaustively enumerates all capture/chain capture moves
 *
 * Historical findings from the original generator:
 * - Some positions on 19x19 and hex boards support capture chains of 10+ segments
 * - 100k+ different capture sequences can achieve maximum segment counts
 *
 * Usage:
 *   npx ts-node scripts/run-chain-capture-exhaustive.ts [options]
 *
 * Options:
 *   --boardTypes=square8,square19,hexagonal  Board types to test (default: all)
 *   --gamesPerBoard=N                        Games to generate per board type (default: 100)
 *   --maxDepth=N                             Max chain depth to explore (default: 20)
 *   --maxSequences=N                         Max sequences per position (default: 100000)
 *   --seed=N                                 Random seed for reproducibility (default: 42)
 *   --verbose                                Enable verbose logging
 *   --output=path                            Output JSON path (default: results/chain_capture_exhaustive.json)
 */

import * as fs from 'fs';
import * as path from 'path';
import type { BoardState, BoardType, Position } from '../src/shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../src/shared/types/game';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters,
} from '../src/client/sandbox/sandboxCaptures';
import {
  applyMarkerEffectsAlongPathOnBoard,
  MarkerPathHelpers,
} from '../src/client/sandbox/sandboxMovement';

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

interface Config {
  boardTypes: BoardType[];
  gamesPerBoard: number;
  maxDepth: number;
  maxSequences: number;
  seed: number;
  verbose: boolean;
  outputPath: string;
}

function parseArgs(): Config {
  const args = process.argv.slice(2);
  const config: Config = {
    boardTypes: ['square8', 'square19', 'hexagonal'],
    gamesPerBoard: 100,
    maxDepth: 20,
    maxSequences: 100000,
    seed: 42,
    verbose: false,
    outputPath: 'results/chain_capture_exhaustive.json',
  };

  for (const arg of args) {
    if (arg.startsWith('--boardTypes=')) {
      config.boardTypes = arg.slice('--boardTypes='.length).split(',') as BoardType[];
    } else if (arg.startsWith('--gamesPerBoard=')) {
      config.gamesPerBoard = parseInt(arg.slice('--gamesPerBoard='.length), 10);
    } else if (arg.startsWith('--maxDepth=')) {
      config.maxDepth = parseInt(arg.slice('--maxDepth='.length), 10);
    } else if (arg.startsWith('--maxSequences=')) {
      config.maxSequences = parseInt(arg.slice('--maxSequences='.length), 10);
    } else if (arg.startsWith('--seed=')) {
      config.seed = parseInt(arg.slice('--seed='.length), 10);
    } else if (arg === '--verbose') {
      config.verbose = true;
    } else if (arg.startsWith('--output=')) {
      config.outputPath = arg.slice('--output='.length);
    } else if (arg === '--help') {
      console.log(`
Chain Capture Exhaustive Generator

Usage:
  npx ts-node scripts/run-chain-capture-exhaustive.ts [options]

Options:
  --boardTypes=square8,square19,hexagonal  Board types to test (default: all)
  --gamesPerBoard=N                        Games to generate per board type (default: 100)
  --maxDepth=N                             Max chain depth to explore (default: 20)
  --maxSequences=N                         Max sequences per position (default: 100000)
  --seed=N                                 Random seed for reproducibility (default: 42)
  --verbose                                Enable verbose logging
  --output=path                            Output JSON path (default: results/chain_capture_exhaustive.json)
`);
      process.exit(0);
    }
  }

  return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// RNG (deterministic LCG)
// ═══════════════════════════════════════════════════════════════════════════

function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Board Creation
// ═══════════════════════════════════════════════════════════════════════════

function createEmptyBoard(boardType: BoardType): BoardState {
  const config = BOARD_CONFIGS[boardType];
  return {
    type: boardType,
    size: config.size,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 0, 2: 0 },
  };
}

function addStack(board: BoardState, pos: Position, player: number, height: number): void {
  const key = positionToString(pos);
  const rings = new Array(height).fill(player);
  board.stacks.set(key, {
    position: pos,
    rings,
    stackHeight: height,
    capHeight: height,
    controllingPlayer: player,
  });
}

function isValidPosition(boardType: BoardType, pos: Position, size: number): boolean {
  if (boardType === 'hexagonal') {
    const x = pos.x;
    const y = pos.y;
    const z = pos.z !== undefined ? pos.z : -x - y;
    const radius = size - 1;
    const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    return dist <= radius;
  }
  return pos.x >= 0 && pos.x < size && pos.y >= 0 && pos.y < size;
}

function generateValidPositions(boardType: BoardType, size: number): Position[] {
  const positions: Position[] = [];

  if (boardType === 'hexagonal') {
    const radius = size - 1;
    for (let x = -radius; x <= radius; x++) {
      for (let y = -radius; y <= radius; y++) {
        const z = -x - y;
        if (Math.abs(z) <= radius) {
          positions.push({ x, y, z });
        }
      }
    }
  } else {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        positions.push({ x, y });
      }
    }
  }

  return positions;
}

// ═══════════════════════════════════════════════════════════════════════════
// Random Board Generation
// ═══════════════════════════════════════════════════════════════════════════

interface RandomBoardConfig {
  attackerHeight: number;
  targetHeight: number;
  additionalStacks: number;
  additionalHeights: number[];
}

// Get position at distance N in a direction from origin
function getPositionAtDistance(
  boardType: BoardType,
  origin: Position,
  direction: Position,
  distance: number,
  size: number
): Position | null {
  let current = origin;
  for (let i = 0; i < distance; i++) {
    const next: Position =
      boardType === 'hexagonal'
        ? {
            x: current.x + direction.x,
            y: current.y + direction.y,
            z: (current.z ?? -current.x - current.y) + (direction.z ?? 0),
          }
        : { x: current.x + direction.x, y: current.y + direction.y };

    if (!isValidPosition(boardType, next, size)) {
      return null;
    }
    current = next;
  }
  return current;
}

// Get all orthogonal directions for a board type
function getDirections(boardType: BoardType): Position[] {
  if (boardType === 'hexagonal') {
    return [
      { x: 1, y: -1, z: 0 },
      { x: 1, y: 0, z: -1 },
      { x: 0, y: 1, z: -1 },
      { x: -1, y: 1, z: 0 },
      { x: -1, y: 0, z: 1 },
      { x: 0, y: -1, z: 1 },
    ];
  }
  return [
    { x: 1, y: 0 },
    { x: -1, y: 0 },
    { x: 0, y: 1 },
    { x: 0, y: -1 },
  ];
}

function generateRandomBoard(
  boardType: BoardType,
  rng: () => number,
  config: RandomBoardConfig
): { board: BoardState; attackerPos: Position } | null {
  const boardConfig = BOARD_CONFIGS[boardType];
  const board = createEmptyBoard(boardType);
  const validPositions = generateValidPositions(boardType, boardConfig.size);
  const usedPositions = new Set<string>();

  // Shuffle positions
  const shuffled = [...validPositions];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  // 1. Place attacker stack (player 1) - choose a position with room for captures
  let attackerPos: Position | null = null;
  const directions = getDirections(boardType);

  for (const candidatePos of shuffled) {
    // Check that this position has at least one valid capture setup
    // (target at distance 1 or 2, landing at distance >= attackerHeight)
    let hasValidCapture = false;
    for (const dir of directions) {
      // Target at distance 1, landing at distance 1+attackerHeight
      const targetPos = getPositionAtDistance(boardType, candidatePos, dir, 1, boardConfig.size);
      const landingPos = getPositionAtDistance(
        boardType,
        candidatePos,
        dir,
        1 + config.attackerHeight,
        boardConfig.size
      );
      if (targetPos && landingPos) {
        hasValidCapture = true;
        break;
      }
    }
    if (hasValidCapture) {
      attackerPos = candidatePos;
      break;
    }
  }

  if (!attackerPos) return null;
  addStack(board, attackerPos, 1, config.attackerHeight);
  usedPositions.add(positionToString(attackerPos));

  // 2. Place target stack (player 2) - guaranteed to be capturable
  //    Place at distance 1 from attacker in a random valid direction
  const shuffledDirs = [...directions];
  for (let i = shuffledDirs.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffledDirs[i], shuffledDirs[j]] = [shuffledDirs[j], shuffledDirs[i]];
  }

  let targetPlaced = false;
  for (const dir of shuffledDirs) {
    const targetPos = getPositionAtDistance(boardType, attackerPos, dir, 1, boardConfig.size);
    const landingPos = getPositionAtDistance(
      boardType,
      attackerPos,
      dir,
      1 + config.attackerHeight,
      boardConfig.size
    );

    if (targetPos && landingPos) {
      const targetKey = positionToString(targetPos);
      const landingKey = positionToString(landingPos);

      if (!usedPositions.has(targetKey) && !usedPositions.has(landingKey)) {
        addStack(board, targetPos, 2, config.targetHeight);
        usedPositions.add(targetKey);
        targetPlaced = true;
        break;
      }
    }
  }

  if (!targetPlaced) return null;

  // 3. Place additional random stacks - try to place some in capturable positions
  //    to create chain opportunities
  let additionalPlaced = 0;
  const targetStackPositions: Position[] = [];

  // First, try to place stacks that extend the capture chain
  for (const stackEntry of board.stacks.values()) {
    if (stackEntry.controllingPlayer === 2) {
      targetStackPositions.push(stackEntry.position);
    }
  }

  // Try to place additional targets along capture paths
  for (
    let attempt = 0;
    attempt < config.additionalStacks * 3 && additionalPlaced < config.additionalStacks;
    attempt++
  ) {
    // Pick a random direction and distance
    const dir = shuffledDirs[Math.floor(rng() * shuffledDirs.length)];
    const basePos =
      targetStackPositions.length > 0
        ? targetStackPositions[Math.floor(rng() * targetStackPositions.length)]
        : attackerPos;

    // Try to place at distance 2-4 from base position (potential landing + new target)
    const distance = 2 + Math.floor(rng() * 3);
    const newPos = getPositionAtDistance(boardType, basePos, dir, distance, boardConfig.size);

    if (newPos) {
      const key = positionToString(newPos);
      if (!usedPositions.has(key)) {
        const height = config.additionalHeights[additionalPlaced % config.additionalHeights.length];
        const player = rng() < 0.7 ? 2 : 1; // Bias toward targets
        addStack(board, newPos, player, height);
        usedPositions.add(key);
        if (player === 2) {
          targetStackPositions.push(newPos);
        }
        additionalPlaced++;
      }
    }
  }

  // Fill remaining with random positions
  for (const pos of shuffled) {
    if (additionalPlaced >= config.additionalStacks) break;
    const key = positionToString(pos);
    if (usedPositions.has(key)) continue;

    const height = config.additionalHeights[additionalPlaced % config.additionalHeights.length];
    const player = rng() < 0.5 ? 1 : 2;
    addStack(board, pos, player, height);
    usedPositions.add(key);
    additionalPlaced++;
  }

  return { board, attackerPos };
}

// ═══════════════════════════════════════════════════════════════════════════
// Capture Sequence Enumeration
// ═══════════════════════════════════════════════════════════════════════════

interface CaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
}

interface CaptureSequence {
  segments: CaptureSegment[];
  finalBoard: BoardState;
}

function cloneBoard(board: BoardState): BoardState {
  return {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };
}

function enumerateSequences(
  boardType: BoardType,
  initialBoard: BoardState,
  from: Position,
  player: number,
  maxDepth: number,
  maxSequences: number
): CaptureSequence[] {
  const sequences: CaptureSequence[] = [];
  const size = initialBoard.size;

  const adapters: CaptureBoardAdapters = {
    isValidPosition: (p: Position) => isValidPosition(boardType, p, size),
    isCollapsedSpace: (p: Position, b: BoardState) => {
      const key = positionToString(p);
      return b.collapsedSpaces.has(key);
    },
    getMarkerOwner: (_p: Position, _b: BoardState) => undefined,
  };

  type Frame = {
    board: BoardState;
    currentPos: Position;
    segments: CaptureSegment[];
    depth: number;
  };

  const stack: Frame[] = [
    {
      board: cloneBoard(initialBoard),
      currentPos: from,
      segments: [],
      depth: 0,
    },
  ];

  while (stack.length > 0 && sequences.length < maxSequences) {
    const frame = stack.pop()!;
    const { board, currentPos, segments, depth } = frame;

    // Depth limit reached - record as maximal sequence
    if (depth >= maxDepth) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    const nextSegments = enumerateCaptureSegmentsFromBoard(
      boardType,
      board,
      currentPos,
      player,
      adapters
    );

    if (nextSegments.length === 0) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    for (const seg of nextSegments) {
      if (sequences.length >= maxSequences) break;

      const boardClone = cloneBoard(board);

      const markerHelpers: MarkerPathHelpers = {
        setMarker: (position, playerNumber, b) => {
          const key = positionToString(position);
          b.markers.set(key, { position, player: playerNumber, type: 'regular' });
        },
        collapseMarker: (position, playerNumber, b) => {
          const key = positionToString(position);
          b.markers.delete(key);
          b.collapsedSpaces.set(key, playerNumber);
        },
        flipMarker: (position, playerNumber, b) => {
          const key = positionToString(position);
          const existing = b.markers.get(key);
          if (existing && existing.player !== playerNumber) {
            b.markers.set(key, { position, player: playerNumber, type: 'regular' });
          }
        },
      };

      const applyAdapters: CaptureApplyAdapters = {
        applyMarkerEffectsAlongPath: (fromPos, toPos, playerNumber) => {
          applyMarkerEffectsAlongPathOnBoard(
            boardClone,
            fromPos,
            toPos,
            playerNumber,
            markerHelpers
          );
        },
      };

      applyCaptureSegmentOnBoard(
        boardClone,
        seg.from,
        seg.target,
        seg.landing,
        player,
        applyAdapters
      );

      stack.push({
        board: boardClone,
        currentPos: seg.landing,
        segments: [...segments, seg],
        depth: depth + 1,
      });
    }
  }

  return sequences;
}

// ═══════════════════════════════════════════════════════════════════════════
// Statistics
// ═══════════════════════════════════════════════════════════════════════════

interface StackInfo {
  position: string;
  player: number;
  height: number;
}

interface SequenceInfo {
  segments: Array<{ from: string; target: string; landing: string }>;
  chainLength: number;
}

interface BoardResult {
  boardIndex: number;
  seed: number;
  boardType: BoardType;
  attackerPos: string;
  stackCount: number;
  stacks: StackInfo[];
  sequenceCount: number;
  maxChainLength: number;
  chainLengthDistribution: Record<number, number>;
  hitMaxSequences: boolean;
  hitMaxDepth: boolean;
  enumerationTimeMs: number;
  // Top sequences for analysis (only included for long chains)
  topSequences: SequenceInfo[] | undefined;
}

interface BoardTypeStats {
  boardType: BoardType;
  totalGames: number;
  totalSequences: number;
  maxSequencesInSingleGame: number;
  maxChainLengthOverall: number;
  avgChainLength: number;
  gamesWithLongChains: number; // chains >= 5
  gamesHitMaxSequences: number;
  gamesHitMaxDepth: number;
  chainLengthDistribution: Record<number, number>;
  topResults: BoardResult[];
}

interface RunSummary {
  config: Config;
  startTime: string;
  endTime: string;
  totalDurationMs: number;
  boardTypeStats: Record<BoardType, BoardTypeStats>;
  overallMaxChainLength: number;
  overallMaxSequences: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  const config = parseArgs();
  const startTime = new Date();
  const rng = makeRng(config.seed);

  console.log('═══════════════════════════════════════════════════════════════════════════');
  console.log('Chain Capture Exhaustive Generator');
  console.log('═══════════════════════════════════════════════════════════════════════════');
  console.log('Configuration:');
  console.log(`  Board types: ${config.boardTypes.join(', ')}`);
  console.log(`  Games per board: ${config.gamesPerBoard}`);
  console.log(`  Max depth: ${config.maxDepth}`);
  console.log(`  Max sequences: ${config.maxSequences}`);
  console.log(`  Seed: ${config.seed}`);
  console.log(`  Output: ${config.outputPath}`);
  console.log('');

  const boardTypeStats: Record<BoardType, BoardTypeStats> = {} as any;
  let overallMaxChainLength = 0;
  let overallMaxSequences = 0;

  for (const boardType of config.boardTypes) {
    console.log(`\n━━━ Processing ${boardType} ━━━`);

    const stats: BoardTypeStats = {
      boardType,
      totalGames: 0,
      totalSequences: 0,
      maxSequencesInSingleGame: 0,
      maxChainLengthOverall: 0,
      avgChainLength: 0,
      gamesWithLongChains: 0,
      gamesHitMaxSequences: 0,
      gamesHitMaxDepth: 0,
      chainLengthDistribution: {},
      topResults: [],
    };

    const allResults: BoardResult[] = [];
    let totalChainLengths = 0;
    let totalSequenceCount = 0;

    for (let i = 0; i < config.gamesPerBoard; i++) {
      const gameSeed = Math.floor(rng() * 0xffffffff);
      const gameRng = makeRng(gameSeed);

      // Generate random board configuration
      const boardConfig: RandomBoardConfig = {
        attackerHeight: 2 + Math.floor(gameRng() * 2), // 2-3
        targetHeight: 2 + Math.floor(gameRng() * 2), // 2-3
        additionalStacks: 2 + Math.floor(gameRng() * 2), // 2-3
        additionalHeights: [2, 3, 2], // mix of heights
      };

      const generated = generateRandomBoard(boardType, gameRng, boardConfig);
      if (!generated) {
        if (config.verbose) {
          console.log(`  Game ${i + 1}: Failed to generate board`);
        }
        continue;
      }

      const { board, attackerPos } = generated;

      const enumStartTime = Date.now();
      const sequences = enumerateSequences(
        boardType,
        board,
        attackerPos,
        1, // player 1 is attacker
        config.maxDepth,
        config.maxSequences
      );
      const enumTimeMs = Date.now() - enumStartTime;

      // Calculate statistics
      const chainLengths = sequences.map((s) => s.segments.length);
      const maxChainLength = chainLengths.length > 0 ? Math.max(...chainLengths) : 0;
      const distribution: Record<number, number> = {};
      for (const len of chainLengths) {
        distribution[len] = (distribution[len] || 0) + 1;
      }

      // Capture stack info for interesting boards
      const stacks: StackInfo[] = [];
      for (const [key, stack] of board.stacks) {
        stacks.push({
          position: key,
          player: stack.controllingPlayer,
          height: stack.stackHeight,
        });
      }

      // Capture top sequences (longest chains) for analysis
      const topSequences: SequenceInfo[] = sequences
        .filter((s) => s.segments.length >= maxChainLength - 1) // Top tier chains
        .slice(0, 10) // Limit to 10
        .map((s) => ({
          segments: s.segments.map((seg) => ({
            from: positionToString(seg.from),
            target: positionToString(seg.target),
            landing: positionToString(seg.landing),
          })),
          chainLength: s.segments.length,
        }));

      const result: BoardResult = {
        boardIndex: i,
        seed: gameSeed,
        boardType,
        attackerPos: positionToString(attackerPos),
        stackCount: board.stacks.size,
        stacks,
        sequenceCount: sequences.length,
        maxChainLength,
        chainLengthDistribution: distribution,
        hitMaxSequences: sequences.length >= config.maxSequences,
        hitMaxDepth: chainLengths.some((len) => len >= config.maxDepth),
        enumerationTimeMs: enumTimeMs,
        topSequences: maxChainLength >= 5 ? topSequences : undefined,
      };

      allResults.push(result);

      // Update stats
      stats.totalGames++;
      stats.totalSequences += sequences.length;
      totalSequenceCount += sequences.length;
      if (sequences.length > stats.maxSequencesInSingleGame) {
        stats.maxSequencesInSingleGame = sequences.length;
      }
      if (maxChainLength > stats.maxChainLengthOverall) {
        stats.maxChainLengthOverall = maxChainLength;
      }
      if (maxChainLength >= 5) {
        stats.gamesWithLongChains++;
      }
      if (result.hitMaxSequences) {
        stats.gamesHitMaxSequences++;
      }
      if (result.hitMaxDepth) {
        stats.gamesHitMaxDepth++;
      }

      for (const [len, count] of Object.entries(distribution)) {
        stats.chainLengthDistribution[parseInt(len)] =
          (stats.chainLengthDistribution[parseInt(len)] || 0) + count;
        totalChainLengths += parseInt(len) * count;
      }

      // Progress logging
      if (config.verbose || (i + 1) % 10 === 0) {
        console.log(
          `  Game ${i + 1}/${config.gamesPerBoard}: ` +
            `${sequences.length} sequences, max chain ${maxChainLength}, ` +
            `${enumTimeMs}ms`
        );
      }
    }

    // Calculate average
    stats.avgChainLength = totalSequenceCount > 0 ? totalChainLengths / totalSequenceCount : 0;

    // Get top results by max chain length
    allResults.sort((a, b) => b.maxChainLength - a.maxChainLength);
    stats.topResults = allResults.slice(0, 10);

    boardTypeStats[boardType] = stats;

    if (stats.maxChainLengthOverall > overallMaxChainLength) {
      overallMaxChainLength = stats.maxChainLengthOverall;
    }
    if (stats.maxSequencesInSingleGame > overallMaxSequences) {
      overallMaxSequences = stats.maxSequencesInSingleGame;
    }

    console.log(`\n${boardType} Summary:`);
    console.log(`  Total games: ${stats.totalGames}`);
    console.log(`  Total sequences: ${stats.totalSequences}`);
    console.log(`  Max sequences in single game: ${stats.maxSequencesInSingleGame}`);
    console.log(`  Max chain length: ${stats.maxChainLengthOverall}`);
    console.log(`  Avg chain length: ${stats.avgChainLength.toFixed(2)}`);
    console.log(`  Games with chains >= 5: ${stats.gamesWithLongChains}`);
    console.log(`  Games hitting max sequences limit: ${stats.gamesHitMaxSequences}`);
    console.log(`  Games hitting max depth limit: ${stats.gamesHitMaxDepth}`);
  }

  const endTime = new Date();
  const summary: RunSummary = {
    config,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    totalDurationMs: endTime.getTime() - startTime.getTime(),
    boardTypeStats,
    overallMaxChainLength,
    overallMaxSequences,
  };

  // Ensure output directory exists
  const outputDir = path.dirname(config.outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Write results
  fs.writeFileSync(config.outputPath, JSON.stringify(summary, null, 2));

  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('Overall Summary');
  console.log('═══════════════════════════════════════════════════════════════════════════');
  console.log(`  Total duration: ${(summary.totalDurationMs / 1000).toFixed(2)}s`);
  console.log(`  Overall max chain length: ${overallMaxChainLength}`);
  console.log(`  Overall max sequences: ${overallMaxSequences}`);
  console.log(`  Results written to: ${config.outputPath}`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
