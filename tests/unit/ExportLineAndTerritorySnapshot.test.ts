import fs from 'fs';
import path from 'path';

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BoardType as BoardTypeAlias,
  GameState,
  Player,
  Position,
  RingStack,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { snapshotFromGameState } from '../utils/stateSnapshots';

/**
 * Utility test to export canonical ComparableSnapshot fixtures for the
 * combined line+territory scenario (FAQ Q7/Q20/Q22/Q23) into JSON files
 * that can be consumed by the Python ai-service parity tests.
 *
 * This mirrors the semantics exercised in
 * `tests/scenarios/LineAndTerritory.test.ts` but focuses on producing
 * a stable post-processing snapshot for each board type.
 *
 * The test is gated by RINGRIFT_EXPORT_PARITY_SNAPSHOTS. Under normal
 * Jest runs it is skipped and has no side effects. To (re)generate the
 * fixtures, run (from the repo root):
 *
 *   RINGRIFT_EXPORT_PARITY_SNAPSHOTS=1 \
 *   npx jest tests/unit/ExportLineAndTerritorySnapshot.test.ts --runInBand
 *
 * Snapshots will be written to:
 *   ai-service/tests/parity/line_territory_scenario_square8.snapshot.json
 *   ai-service/tests/parity/line_territory_scenario_square19.snapshot.json
 *   ai-service/tests/parity/line_territory_scenario_hexagonal.snapshot.json
 */

const EXPORT_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_EXPORT_PARITY_SNAPSHOTS ?? '');

const maybeTest = EXPORT_ENABLED ? test : test.skip;

const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

const basePlayers: Player[] = [
  {
    id: 'p1',
    username: 'Player1',
    type: 'human',
    playerNumber: 1,
    isReady: true,
    timeRemaining: timeControl.initialTime * 1000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
  },
  {
    id: 'p2',
    username: 'Player2',
    type: 'human',
    playerNumber: 2,
    isReady: true,
    timeRemaining: timeControl.initialTime * 1000,
    ringsInHand: 18,
    eliminatedRings: 0,
    territorySpaces: 0,
  },
];

function createEngine(boardType: BoardType): {
  engine: GameEngine;
  gameState: GameState;
  boardManager: any;
} {
  const engine = new GameEngine(
    'export-line-territory',
    boardType as BoardTypeAlias,
    basePlayers,
    timeControl,
    false
  );
  const engineAny: any = engine;
  const gameState: GameState = engineAny.gameState as GameState;
  const boardManager: any = engineAny.boardManager;
  return { engine, gameState, boardManager };
}

function makeStack(
  boardManager: any,
  gameState: GameState,
  playerNumber: number,
  height: number,
  position: Position
) {
  const rings = Array(height).fill(playerNumber);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer: playerNumber,
  };
  boardManager.setStack(position, stack, gameState.board);
}

const boardTypesUnderTest: BoardType[] = ['square8', 'square19', 'hexagonal'];

maybeTest('export line+territory scenario snapshots for all board types', async () => {
  const outDir = path.join(process.cwd(), 'ai-service', 'tests', 'parity');
  fs.mkdirSync(outDir, { recursive: true });

  for (const boardType of boardTypesUnderTest) {
    const { engine, gameState, boardManager } = createEngine(boardType);
    const engineAny: any = engine;
    const board = gameState.board;
    const requiredLength = BOARD_CONFIGS[gameState.boardType].lineLength;

    gameState.currentPlayer = 1;

    // Clear any existing board state for a clean scenario.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Synthetic overlong line (length = requiredLength + 1) for Player 1.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      linePositions.push({ x: i, y: 0 });
    }

    // Stub BoardManager.findAllLines to return this single line once,
    // then no further lines, mirroring LineAndTerritory.test.ts.
    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy
      .mockImplementationOnce(() => [
        {
          player: 1,
          positions: linePositions,
          length: linePositions.length,
          direction: { x: 1, y: 0 },
        },
      ])
      .mockImplementation(() => []);

    // Territory region: a single-cell region at (5,5) containing a
    // Player 2 stack, plus a Player 1 stack outside the region so the
    // self-elimination prerequisite is satisfied.
    const regionPos: Position = { x: 5, y: 5 };
    const outsidePos: Position = { x: 7, y: 7 };

    makeStack(boardManager, gameState, 2, 1, regionPos); // victim inside region
    makeStack(boardManager, gameState, 1, 2, outsidePos); // P1 stack outside

    const regionTerritory = {
      spaces: [regionPos],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    const initialTerritory = gameState.players.find((p) => p.playerNumber === 1)!.territorySpaces;
    const initialEliminated = gameState.players.find((p) => p.playerNumber === 1)!.eliminatedRings;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialCollapsedCount = board.collapsedSpaces.size;

    // 1) Process line formations for the current player (Player 1).
    await engineAny.processLineFormations();

    const afterLinesState: GameState = engine.getGameState();
    const player1AfterLines = afterLinesState.players.find((p) => p.playerNumber === 1)!;

    // Sanity-check the expected Option 2 behaviour to avoid exporting
    // an unexpected snapshot shape.
    const collapsedKeysAfterLines = new Set<string>();
    for (const [key, owner] of afterLinesState.board.collapsedSpaces) {
      if (owner === 1) collapsedKeysAfterLines.add(key);
    }

    if (collapsedKeysAfterLines.size - initialCollapsedCount !== requiredLength) {
      throw new Error(
        `Unexpected collapsed-space delta for boardType=${boardType}: ` +
          `expected ${requiredLength}, got ${collapsedKeysAfterLines.size - initialCollapsedCount}`
      );
    }
    if (player1AfterLines.eliminatedRings !== initialEliminated) {
      throw new Error('Line processing eliminated rings; expected 0 elimination in Option 2 path');
    }
    if (afterLinesState.totalRingsEliminated !== initialTotalEliminated) {
      throw new Error('Total rings eliminated changed during line processing; expected no change');
    }

    // 2) Territory processing via the move-driven decision model.
    engineAny.useMoveDrivenDecisionPhases = true;
    engineAny.gameState.currentPhase = 'territory_processing';

    const territoryMoves = engineAny.getValidTerritoryProcessingMoves(1);
    if (territoryMoves.length > 0) {
      const regionMove = territoryMoves[0];
      await engine.makeMove(regionMove);

      const elimMoves = engine.getValidMoves(1);
      const elimMove = elimMoves.find((m: any) => m.type === 'eliminate_rings_from_stack');
      if (elimMove) {
        await engine.makeMove(elimMove);
      }
    }

    const finalState: GameState = engine.getGameState();
    const player1Final = finalState.players.find((p) => p.playerNumber === 1)!;

    const regionKey = positionToString(regionPos);
    if (finalState.board.stacks.get(regionKey) !== undefined) {
      throw new Error('Expected region stack to be eliminated after territory processing');
    }

    // Ensure we still have a territory increment for the region and
    // a non-zero elimination delta for Player 1; exact values are
    // asserted in the dedicated scenario tests.
    if (
      player1Final.territorySpaces <
      initialTerritory + regionTerritory.spaces.length
    ) {
      throw new Error('Unexpected territorySpaces delta after territory processing');
    }
    if (finalState.totalRingsEliminated <= initialTotalEliminated) {
      throw new Error('Expected at least one ring elimination from territory processing');
    }

    const snapshotLabel = `line_territory_${boardType}`;
    const snapshot = snapshotFromGameState(snapshotLabel, finalState);
    const outPath = path.join(
      outDir,
      `line_territory_scenario_${boardType}.snapshot.json`
    );
    fs.writeFileSync(outPath, JSON.stringify(snapshot, null, 2), 'utf8');

    findAllLinesSpy.mockRestore();
    findDisconnectedRegionsSpy.mockRestore();
  }
});

