/*
 * Orchestrator-focused contract vector generator.
 *
 * Generates multi-step v2 contract vectors for composite territory
 * scenarios (region processing + self-elimination) using the canonical
 * shared orchestrator (processTurn).
 */

import fs from 'fs';
import path from 'path';

import type {
  BoardType,
  TimeControl,
  Player,
  GameState,
  Position,
  Territory,
  Move,
} from '../src/shared/types/game';
import { positionToString } from '../src/shared/types/game';

import { createInitialGameState } from '../src/shared/engine/initialState';
import { processTurn } from '../src/shared/engine/orchestration/turnOrchestrator';
import {
  createContractTestVector,
  exportVectorBundle,
  type ContractTestVector,
} from '../src/shared/engine/contracts/testVectorGenerator';
import { enumerateTerritoryEliminationMoves } from '../src/shared/engine/territoryDecisionHelpers';

const BOARD_TYPE: BoardType = 'square8';
const TIME_CONTROL: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

const BASE_PLAYERS: Player[] = [
  {
    id: 'p1',
    username: 'Player 1',
    type: 'human',
    playerNumber: 1,
    isReady: true,
    timeRemaining: TIME_CONTROL.initialTime * 1000,
    ringsInHand: 0,
    eliminatedRings: 0,
    territorySpaces: 0,
  },
  {
    id: 'p2',
    username: 'Player 2',
    type: 'human',
    playerNumber: 2,
    isReady: true,
    timeRemaining: TIME_CONTROL.initialTime * 1000,
    ringsInHand: 0,
    eliminatedRings: 0,
    territorySpaces: 0,
  },
];

function createEmptyTerritoryState(gameId: string): GameState {
  const initial = createInitialGameState(
    gameId,
    BOARD_TYPE,
    BASE_PLAYERS,
    TIME_CONTROL,
    false
  ) as unknown as GameState;

  initial.currentPlayer = 1;
  initial.currentPhase = 'territory_processing';
  initial.gameStatus = 'active';

  initial.board.stacks.clear();
  initial.board.markers.clear();
  initial.board.collapsedSpaces.clear();
  initial.board.territories = new Map();
  initial.board.formedLines = [];
  initial.board.eliminatedRings = { 1: 0, 2: 0 };

  initial.totalRingsEliminated = 0;
  initial.players = initial.players.map((p) => ({
    ...p,
    ringsInHand: 0,
    eliminatedRings: 0,
    territorySpaces: 0,
  }));

  return initial;
}

interface TerritoryCompositeSequence {
  initialState: GameState;
  regionMove: Move;
  stateAfterRegion: GameState;
  eliminationMove: Move;
  stateAfterElimination: GameState;
}

function buildTerritoryRegionSelfEliminationSequence(): TerritoryCompositeSequence {
  const state = createEmptyTerritoryState('contract-territory-composite');
  const board = state.board;

  const p1a: Position = { x: 0, y: 0 };
  const p2a: Position = { x: 1, y: 0 };
  const outside: Position = { x: 3, y: 3 };

  const p1aKey = positionToString(p1a);
  const p2aKey = positionToString(p2a);
  const outsideKey = positionToString(outside);

  // Create a region controlled by player 1 (both stacks controlled by P1)
  // This forms a disconnected region that triggers self-elimination
  board.stacks.set(p1aKey, {
    position: p1a,
    rings: [1, 1],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: 1,
  } as any);

  board.stacks.set(p2aKey, {
    position: p2a,
    rings: [1, 1, 1],
    stackHeight: 3,
    capHeight: 3,
    controllingPlayer: 1,
  } as any);

  board.stacks.set(outsideKey, {
    position: outside,
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 1,
  } as any);

  const region: Territory = {
    spaces: [p1a, p2a],
    controllingPlayer: 1,
    isDisconnected: true,
  };

  const regionMove: Move = {
    id: 'territory-composite-region',
    type: 'process_territory_region',
    player: 1,
    to: p1a,
    disconnectedRegions: [region],
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  } as any;

  const regionResult = processTurn(state, regionMove);
  const stateAfterRegion = regionResult.nextState;

  // Check if game ended after region processing (e.g., victory condition met)
  if (
    stateAfterRegion.gameStatus === 'completed' ||
    stateAfterRegion.currentPhase === 'game_over'
  ) {
    // No self-elimination step needed - game already ended
    return {
      initialState: state,
      regionMove,
      stateAfterRegion,
      eliminationMove: null as unknown as Move,
      stateAfterElimination: stateAfterRegion,
    };
  }

  const eliminationCandidates = enumerateTerritoryEliminationMoves(stateAfterRegion, 1);
  if (eliminationCandidates.length === 0) {
    // No self-elimination needed
    return {
      initialState: state,
      regionMove,
      stateAfterRegion,
      eliminationMove: null as unknown as Move,
      stateAfterElimination: stateAfterRegion,
    };
  }

  const eliminationCandidate =
    eliminationCandidates.find((m) => m.to && positionToString(m.to) === outsideKey) ||
    eliminationCandidates[0];

  const eliminationMove = eliminationCandidate as unknown as Move;
  const eliminationResult = processTurn(stateAfterRegion, eliminationMove);
  const stateAfterElimination = eliminationResult.nextState;

  return {
    initialState: state,
    regionMove,
    stateAfterRegion,
    eliminationMove,
    stateAfterElimination,
  };
}

function buildTerritoryProcessingVectors(): ContractTestVector[] {
  const sequence = buildTerritoryRegionSelfEliminationSequence();
  const vectors: ContractTestVector[] = [];

  const sequenceTag = 'sequence:territory.region.self_elim.square8';

  const v1 = createContractTestVector(
    sequence.initialState,
    sequence.regionMove,
    sequence.stateAfterRegion,
    {
      description: 'Territory region processing (two-space region, square8)',
      source: 'generated',
      tags: ['territory', 'territory_processing', 'orchestrator', 'parity', sequenceTag],
    }
  );
  v1.expectedOutput.status = 'complete';
  vectors.push(v1);

  // Only add elimination vector if there's an elimination move
  if (sequence.eliminationMove) {
    const v2 = createContractTestVector(
      sequence.stateAfterRegion,
      sequence.eliminationMove,
      sequence.stateAfterElimination,
      {
        description: 'Territory self-elimination after region processing (square8)',
        source: 'generated',
        tags: ['territory', 'territory_processing', 'orchestrator', 'parity', sequenceTag],
      }
    );
    v2.expectedOutput.status = 'complete';
    vectors.push(v2);
  }

  return vectors;
}

function writeBundle(fileName: string, vectors: ContractTestVector[]): void {
  const outDir = path.resolve(__dirname, '../tests/fixtures/contract-vectors/v2');
  const outPath = path.join(outDir, fileName);
  const json = exportVectorBundle(vectors);
  fs.writeFileSync(outPath, json, 'utf8');
  // eslint-disable-next-line no-console
  console.log(`Wrote ${vectors.length} vectors to ${outPath}`);
}

async function main(): Promise<void> {
  const territoryVectors = buildTerritoryProcessingVectors();
  writeBundle('territory_processing.vectors.json', territoryVectors);
}

main().catch((err) => {
  console.error('Error generating orchestrator contract vectors:', err);
  process.exitCode = 1;
});
