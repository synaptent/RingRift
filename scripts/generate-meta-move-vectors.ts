/*
 * Meta-move contract vector generator (swap_sides / pie rule).
 *
 * Generates a small v2 bundle focused on the swap_sides meta-move using
 * the backend GameEngine host. This is kept separate from
 * generate-extended-contract-vectors.ts so that meta-move coverage does
 * not depend on any orchestrator/territory aggregates.
 */

import fs from 'fs';
import path from 'path';

import type { BoardType, TimeControl, Player, GameState, Move } from '../src/shared/types/game';
import {
  createContractTestVector,
  exportVectorBundle,
  type ContractTestVector,
} from '../src/shared/engine/contracts/testVectorGenerator';
import { GameEngine } from '../src/server/game/GameEngine';

const BOARD_TYPE: BoardType = 'square8';
const TIME_CONTROL: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

const BASE_PLAYERS: Player[] = [
  {
    id: 'p1',
    username: 'Player 1',
    type: 'ai',
    playerNumber: 1,
    isReady: true,
    timeRemaining: TIME_CONTROL.initialTime * 1000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
  },
  {
    id: 'p2',
    username: 'Player 2',
    type: 'ai',
    playerNumber: 2,
    isReady: true,
    timeRemaining: TIME_CONTROL.initialTime * 1000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
  },
];

function writeBundle(fileName: string, vectors: ContractTestVector[]): void {
  const outDir = path.resolve(__dirname, '../tests/fixtures/contract-vectors/v2');
  const outPath = path.join(outDir, fileName);
  const json = exportVectorBundle(vectors);
  fs.writeFileSync(outPath, json, 'utf8');
  // eslint-disable-next-line no-console
  console.log(`Wrote ${vectors.length} vectors to ${outPath}`);
}

async function createMetaMoveVectors(): Promise<ContractTestVector[]> {
  const vectors: ContractTestVector[] = [];

  const engine = new GameEngine(
    'contract-meta-move-swap-sides',
    BOARD_TYPE,
    BASE_PLAYERS,
    TIME_CONTROL,
    false
  );
  const started = engine.startGame();
  if (!started) {
    throw new Error('Failed to start GameEngine for swap_sides meta-move vector');
  }

  const engineAny: any = engine;
  engineAny.gameState.rulesOptions = {
    ...(engineAny.gameState.rulesOptions || {}),
    swapRuleEnabled: true,
  };

  // P1 opening: place a single ring.
  const openingMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
    type: 'place_ring',
    player: 1,
    to: { x: 3, y: 3 },
    placementCount: 1,
  } as any;

  const openingResult = await engine.makeMove(openingMove);
  if (!openingResult.success) {
    throw new Error(
      `Opening move for swap_sides vector failed: ${openingResult.error ?? 'unknown error'}`
    );
  }

  // P2 swap_sides meta-move.
  // Ensure the backend state reflects that it is now Player 2's turn in an
  // active interactive phase. In practice, makeMove() should already have
  // advanced to P2; this assignment is a defensive guard to keep the
  // contract vector focused on swap_sides semantics rather than host
  // turn-advancement details.
  engineAny.gameState.currentPlayer = 2;

  const beforeSwap = engine.getGameState() as GameState;
  const swapMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
    type: 'swap_sides',
    player: 2,
    to: { x: 0, y: 0 },
  } as any;

  const swapResult = await engine.makeMove(swapMove);
  if (!swapResult.success || !swapResult.gameState) {
    throw new Error(`swap_sides meta-move failed: ${swapResult.error ?? 'unknown error'}`);
  }

  const afterSwap = swapResult.gameState as GameState;

  const vector = createContractTestVector(beforeSwap, swapMove as Move, afterSwap, {
    description: 'Pie-rule swap_sides meta-move immediately after P1 opening on square8',
    tags: ['swap_sides', 'pie_rule'],
    source: 'generated',
  });

  // Stamp stable ID/category/tags for SSOT and Python parity diagnostics.
  vector.id = 'meta.swap_sides.after_p1_first_move.square8';
  vector.category = 'edge_case';
  vector.description = 'swap_sides meta-move (square8, after P1 first turn)';
  vector.expectedOutput.status = 'complete';

  const baseTags = vector.tags ?? [];
  const tags = new Set<string>(['swap_sides', 'pie_rule', 'meta_move', 'edge_case', ...baseTags]);
  vector.tags = Array.from(tags);

  vectors.push(vector);
  return vectors;
}

async function main(): Promise<void> {
  const vectors = await createMetaMoveVectors();
  writeBundle('meta_moves.vectors.json', vectors);
}

main().catch((err) => {
  console.error('Error generating meta-move contract vectors:', err);
  process.exitCode = 1;
});
