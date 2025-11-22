import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  BoardType,
  Player,
  TimeControl,
  Move,
  Position,
  GameState as SharedGameState,
  GamePhase,
  GameStatus,
  LineInfo,
  Territory,
} from '../../src/shared/types/game';
import { GameEngine } from '../../src/shared/engine/GameEngine';
import {
  validatePlacement,
  validateSkipPlacement,
} from '../../src/shared/engine/validators/PlacementValidator';
import { validateMovement } from '../../src/shared/engine/validators/MovementValidator';
import { validateCapture } from '../../src/shared/engine/validators/CaptureValidator';
import {
  validateProcessLine,
  validateChooseLineReward,
} from '../../src/shared/engine/validators/LineValidator';
import {
  validateProcessTerritory,
  validateEliminateStack,
} from '../../src/shared/engine/validators/TerritoryValidator';
import {
  GameState as EngineGameState,
  PlaceRingAction,
  MoveStackAction,
  OvertakingCaptureAction,
  ContinueChainAction,
  ProcessLineAction,
  ChooseLineRewardAction,
  ProcessTerritoryAction,
  EliminateStackAction,
  SkipPlacementAction,
  ValidationResult,
} from '../../src/shared/engine/types';
import {
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../src/shared/engine/core';
import { moveToGameAction } from '../../src/shared/engine/moveActionAdapter';

/**
 * TS→Python rules parity fixture generator (v1 + v2).
 *
 * This script uses the shared TypeScript rules engine to emit canonical
 * fixtures that Python parity tests can consume. Fixtures are written
 * under tests/fixtures/rules-parity/ as JSON.
 *
 * v1 coverage (square8 only):
 * - A single square8 / 2-player initial state (state-only).
 * - A simple placement scenario: Player 1 placing a single ring at (3,3).
 * - A simple movement scenario: the same stack moving from (3,3) to (3,4).
 * - Overtaking capture
 * - Chain capture continuation
 * - Line processing (single line)
 * - Line reward choice (minimum collapse + collapse-all)
 * - Territory processing (single disconnected region)
 * - Ring elimination
 * - Skip placement (valid + invalid no-stacks case)
 *
 * v2 coverage (square8, square19, hexagonal):
 * - Multi-line line-processing scenarios exercising:
 *   - Multiple simultaneous lines, including overlapping coordinates.
 *   - Mixed-player lines on the same board.
 *   - Non-trivial lineIndex selection (processing the second line).
 * - Multi-region territory-processing scenarios with:
 *   - Multiple regions, including disconnected ones.
 *   - Regions controlled by both players.
 *   - Off-region stacks for both players to pay territory costs.
 *
 * For the state+action fixtures we encode, in `expected`:
 * - `tsValid`: the shared-engine validator verdict.
 * - `tsValidation`: full ValidationResult for diagnostics.
 * - `tsNext`: a compact snapshot of the post-move GameState as seen by
 *   the shared engine (phase/status, board summary, S-invariant, hash).
 */

type FixtureVersion = 'v1' | 'v2';

interface StateOnlyFixture {
  id: string;
  boardType: BoardType;
  state: unknown;
}

interface OutcomeSnapshot {
  boardType: BoardType;
  currentPhase: GamePhase;
  currentPlayer: number;
  gameStatus: GameStatus;
  totalRingsInPlay: number;
  totalRingsEliminated: number;
  victoryThreshold: number;
  territoryVictoryThreshold: number;
  markers: number;
  collapsed: number;
  eliminated: number;
  S: number;
  stateHash: string;
}

interface StateActionExpected {
  /** Whether the canonical TS validator for this action considers it legal. */
  tsValid: boolean;
  /** Full ValidationResult payload for richer diagnostics. */
  tsValidation: ValidationResult;
  /** Snapshot of the post-move shared-engine state, when applicable. */
  tsNext?: OutcomeSnapshot;
}

interface StateActionFixture {
  id: string;
  version: FixtureVersion;
  boardType: BoardType;
  description: string;
  state: SharedGameState;
  move: Move;
  expected: StateActionExpected;
}

/**
 * Multi-step trace fixtures (TS -> Python).
 *
 * These are used to capture a sequence of canonical Moves, along with
 * optional TS-side expectations (validator verdicts, state hash, S-invariant)
 * after each step. Python parity tests replay these traces via the Python
 * GameEngine and compare hashes / S-invariants against the TS values.
 */
interface TraceStepExpected {
  tsValid?: boolean;
  tsStateHash?: string | undefined;
  tsS?: number | undefined;
}

interface TraceStep {
  label?: string;
  move: Move;
  expected?: TraceStepExpected;
  stateHash?: string | undefined;
  sInvariant?: number | undefined;
}

interface TraceFixture {
  version: FixtureVersion;
  boardType: BoardType;
  initialState: SharedGameState;
  steps: TraceStep[];
}

function makeTraceFromStateAction(
  boardType: BoardType,
  fixture: StateActionFixture,
  label: string
): TraceFixture {
  const { expected } = fixture;
  return {
    version: fixture.version,
    boardType,
    initialState: fixture.state,
    steps: [
      {
        label,
        move: fixture.move,
        expected: {
          tsValid: expected.tsValid,
          tsStateHash: expected.tsNext?.stateHash,
          tsS: expected.tsNext?.S,
        },
        stateHash: expected.tsNext?.stateHash,
        sInvariant: expected.tsNext?.S,
      },
    ],
  };
}

function makePlayers(numPlayers: number): Player[] {
  const players: Player[] = [];
  for (let i = 0; i < numPlayers; i += 1) {
    players.push({
      id: `p${i + 1}`,
      username: `Player ${i + 1}`,
      type: 'human',
      playerNumber: i + 1,
      isReady: true,
      timeRemaining: 600_000,
      // ringsInHand / eliminatedRings / territorySpaces will be populated
      // by createInitialGameState according to BOARD_CONFIGS.
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    });
  }
  return players;
}

function makeTimeControl(): TimeControl {
  return {
    initialTime: 600,
    increment: 5,
    type: 'blitz',
  };
}

function toSharedState(engineState: EngineGameState, boardType: BoardType): SharedGameState {
  // The engine GameState is structurally compatible with the shared
  // GameState shape except for `boardType`, which we inject here so that
  // Python can hydrate the JSON into its Pydantic GameState model.
  return {
    boardType,
    ...(engineState as unknown as Omit<SharedGameState, 'boardType'>),
  };
}

/**
 * Convert an engine GameState into a JSON-friendly SharedGameState where all
 * Map-based board collections (stacks, markers, collapsedSpaces, territories)
 * are represented as plain objects. This is what the Python Pydantic models
 * expect when hydrating GameState from fixture JSON.
 */
function toFixtureState(engineState: EngineGameState, boardType: BoardType): SharedGameState {
  const shared = toSharedState(engineState, boardType) as any;
  const board = engineState.board as any;

  const stacks: Record<string, unknown> = {};
  board.stacks.forEach((value: unknown, key: string) => {
    stacks[key] = value;
  });

  const markers: Record<string, unknown> = {};
  board.markers.forEach((value: unknown, key: string) => {
    markers[key] = value;
  });

  const collapsedSpaces: Record<string, unknown> = {};
  board.collapsedSpaces.forEach((value: unknown, key: string) => {
    collapsedSpaces[key] = value;
  });

  const territories: Record<string, unknown> = {};
  board.territories.forEach((value: unknown, key: string) => {
    territories[key] = value;
  });

  return {
    ...shared,
    board: {
      ...board,
      stacks,
      markers,
      collapsedSpaces,
      territories,
    },
  } as SharedGameState;
}

function snapshotState(state: SharedGameState): OutcomeSnapshot {
  const progress = computeProgressSnapshot(state as any);
  const summary = summarizeBoard(state.board as any);
  return {
    boardType: state.boardType,
    currentPhase: state.currentPhase,
    currentPlayer: state.currentPlayer,
    gameStatus: state.gameStatus,
    totalRingsInPlay: state.totalRingsInPlay,
    totalRingsEliminated: state.totalRingsEliminated,
    victoryThreshold: state.victoryThreshold,
    territoryVictoryThreshold: state.territoryVictoryThreshold,
    markers: summary.markers.length,
    collapsed: summary.collapsedSpaces.length,
    eliminated: progress.eliminated,
    S: progress.S,
    stateHash: hashGameState(state as any),
  };
}

function generateSquare8TwoPlayerInitial(engineState?: EngineGameState): StateOnlyFixture {
  const boardType: BoardType = 'square8';

  let baseState: EngineGameState;
  if (engineState) {
    baseState = engineState;
  } else {
    const players = makePlayers(2);
    const timeControl = makeTimeControl();
    baseState = createInitialGameState(
      'rules-parity-square8-2p',
      boardType,
      players,
      timeControl,
      false
    );
  }

  const stateWithMeta = toFixtureState(baseState, boardType);

  return {
    id: 'square8_2p_initial',
    boardType,
    state: stateWithMeta,
  };
}

function generatePlacementFixture(engineState: EngineGameState): {
  fixture: StateActionFixture;
  engineAfter: EngineGameState;
} {
  const boardType: BoardType = 'square8';
  const fixtureBefore = toFixtureState(engineState, boardType);

  // Canonical simple placement: Player 1 places a single ring at (3,3).
  const to: Position = { x: 3, y: 3 };

  const move: Move = {
    id: 'fixture-square8-2p-place-center',
    type: 'place_ring',
    player: 1,
    to,
    placedOnStack: false,
    placementCount: 1,
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: 1,
  };

  const action: PlaceRingAction = {
    type: 'PLACE_RING',
    playerId: move.player,
    position: to,
    count: move.placementCount ?? 1,
  };

  const tsValidation = validatePlacement(engineState, action);

  // Apply via shared GameEngine to obtain the canonical post-move state.
  const engine = new GameEngine(engineState);
  const event = engine.processAction(action);
  if (event.type !== 'ACTION_PROCESSED') {
    // If something goes wrong, we still emit the fixture but without
    // a tsNext snapshot so Python can at least assert validation parity.
    return {
      fixture: {
        id: 'square8_2p_initial.place_ring_center',
        version: 'v1',
        boardType,
        description:
          'Initial square8 / 2-player state with Player 1 placing a single ring at (3,3).',
        state: fixtureBefore,
        move,
        expected: {
          tsValid: tsValidation.valid,
          tsValidation,
        },
      },
      engineAfter: engineState,
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  const expected: StateActionExpected = {
    tsValid: tsValidation.valid,
    tsValidation,
    tsNext: snapshotState(sharedAfter),
  };

  return {
    fixture: {
      id: 'square8_2p_initial.place_ring_center',
      version: 'v1',
      boardType,
      description: 'Initial square8 / 2-player state with Player 1 placing a single ring at (3,3).',
      state: fixtureBefore,
      move,
      expected,
    },
    engineAfter,
  };
}

function generateMoveStackFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Ensure we have a shared-state view and that the board actually has
  // a stack at (3,3) from the previous placement fixture.
  const sharedBefore = toSharedState(engineState, boardType);
  const fixtureBefore = toFixtureState(engineState, boardType);

  const from: Position = { x: 3, y: 3 };
  const to: Position = { x: 3, y: 4 };

  const move: Move = {
    id: 'fixture-square8-2p-move-stack',
    type: 'move_stack',
    player: sharedBefore.currentPlayer,
    from,
    to,
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: sharedBefore.moveHistory.length + 1,
  };

  // Use the adapter so that this fixture exercises the same mapping
  // path as the production adapter does.
  const action = moveToGameAction(move, engineState) as MoveStackAction;

  const tsValidation = validateMovement(engineState, action);

  // Apply via shared GameEngine to obtain the canonical post-move state.
  const engine = new GameEngine(engineState);
  const event = engine.processAction(action);
  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p_afterPlacement.move_stack_forward',
      version: 'v1',
      boardType,
      description: 'After placing at (3,3), Player 1 attempts to move that stack to (3,4).',
      state: fixtureBefore,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  const expected: StateActionExpected = {
    tsValid: tsValidation.valid,
    tsValidation,
    tsNext: snapshotState(sharedAfter),
  };

  return {
    id: 'square8_2p_afterPlacement.move_stack_forward',
    version: 'v1',
    boardType,
    description: 'After placing at (3,3), Player 1 moves that stack one step forward to (3,4).',
    state: fixtureBefore,
    move,
    expected,
  };
}

function generateOvertakingCaptureFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 has a stack of 2 at (3,3), Player 2 has a stack of 1 at (3,4).
  // Player 1 moves (3,3) -> (3,5), capturing (3,4).

  // Manually construct state for this scenario
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'movement';

  // Clear board stacks
  state.board.stacks = new Map();
  // Ensure maps are initialized (JSON.parse/stringify converts Maps to objects or arrays)
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  // Player 1 stack at (3,3), height 2
  state.board.stacks.set('3,3', {
    position: { x: 3, y: 3 },
    rings: [1, 1],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: 1,
  });

  // Player 2 stack at (3,4), height 1
  state.board.stacks.set('3,4', {
    position: { x: 3, y: 4 },
    rings: [2],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 2,
  });

  const from: Position = { x: 3, y: 3 };
  const captureTarget: Position = { x: 3, y: 4 };
  const to: Position = { x: 3, y: 5 };

  const move: Move = {
    id: 'fixture-square8-2p-overtaking-capture',
    type: 'overtaking_capture',
    player: 1,
    from,
    captureTarget,
    to,
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: OvertakingCaptureAction = {
    type: 'OVERTAKING_CAPTURE',
    playerId: 1,
    from,
    captureTarget,
    to,
  };

  const tsValidation = validateCapture(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.overtaking_capture',
      version: 'v1',
      boardType,
      description: 'Player 1 captures Player 2 stack.',
      state: toFixtureState(state as EngineGameState, boardType),
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.overtaking_capture',
    version: 'v1',
    boardType,
    description: 'Player 1 captures Player 2 stack.',
    state: toFixtureState(state as EngineGameState, boardType),
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateContinueCaptureFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 is in chain_capture phase at (3,5).
  // Another Player 2 stack at (3,6).
  // Player 1 captures (3,6) -> (3,7).

  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'chain_capture';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  // Player 1 stack at (3,5), height 2 (manually adjusted for valid continuation distance)
  state.board.stacks.set('3,5', {
    position: { x: 3, y: 5 },
    rings: [1, 2],
    stackHeight: 2,
    capHeight: 1,
    controllingPlayer: 1,
  });

  // Player 2 stack at (3,6), height 1
  state.board.stacks.set('3,6', {
    position: { x: 3, y: 6 },
    rings: [2],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 2,
  });

  const from: Position = { x: 3, y: 5 };
  const captureTarget: Position = { x: 3, y: 6 };
  const to: Position = { x: 3, y: 7 };

  const move: Move = {
    id: 'fixture-square8-2p-continue-capture',
    type: 'continue_capture_segment',
    player: 1,
    from,
    captureTarget,
    to,
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ContinueChainAction = {
    type: 'CONTINUE_CHAIN',
    playerId: 1,
    from,
    captureTarget,
    to,
  };

  // Reuse validateCapture for chain segments as per engine logic
  const tsValidation = validateCapture(state, action as unknown as OvertakingCaptureAction);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.continue_capture',
      version: 'v1',
      boardType,
      description: 'Player 1 continues capture chain.',
      state: toFixtureState(state as EngineGameState, boardType),
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.continue_capture',
    version: 'v1',
    boardType,
    description: 'Player 1 continues capture chain.',
    state: toFixtureState(state as EngineGameState, boardType),
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateProcessLineFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 has a line of 4 markers.
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'line_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  const line: LineInfo = {
    positions: [
      { x: 0, y: 0 },
      { x: 0, y: 1 },
      { x: 0, y: 2 },
      { x: 0, y: 3 },
    ],
    player: 1,
    length: 4,
    direction: { x: 0, y: 1 },
  };

  state.board.formedLines = [line];

  const move: Move = {
    id: 'fixture-square8-2p-process-line',
    type: 'process_line',
    player: 1,
    to: { x: 0, y: 0 }, // Sentinel
    formedLines: [line],
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ProcessLineAction = {
    type: 'PROCESS_LINE',
    playerId: 1,
    lineIndex: 0,
  };

  const tsValidation = validateProcessLine(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.process_line',
      version: 'v1',
      boardType,
      description: 'Player 1 processes a line.',
      state: toFixtureState(state as EngineGameState, boardType),
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.process_line',
    version: 'v1',
    boardType,
    description: 'Player 1 processes a line.',
    state: toFixtureState(state as EngineGameState, boardType),
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateChooseLineRewardFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 has a line of 5 markers (allowing choice).
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'line_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  const line: LineInfo = {
    positions: [
      { x: 0, y: 0 },
      { x: 0, y: 1 },
      { x: 0, y: 2 },
      { x: 0, y: 3 },
      { x: 0, y: 4 },
    ],
    player: 1,
    length: 5,
    direction: { x: 0, y: 1 },
  };

  state.board.formedLines = [line];

  // Choose Option 2: Minimum collapse (4 markers)
  const collapsedMarkers = [
    { x: 0, y: 0 },
    { x: 0, y: 1 },
    { x: 0, y: 2 },
    { x: 0, y: 3 },
  ];

  const move: Move = {
    id: 'fixture-square8-2p-choose-line-reward',
    type: 'choose_line_reward',
    player: 1,
    to: { x: 0, y: 0 }, // Sentinel
    formedLines: [line],
    collapsedMarkers,
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ChooseLineRewardAction = {
    type: 'CHOOSE_LINE_REWARD',
    playerId: 1,
    lineIndex: 0,
    selection: 'MINIMUM_COLLAPSE',
    collapsedPositions: collapsedMarkers,
  };

  const tsValidation = validateChooseLineReward(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.choose_line_reward',
      version: 'v1',
      boardType,
      description: 'Player 1 chooses minimum collapse for line of 5.',
      state: toFixtureState(state as EngineGameState, boardType),
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.choose_line_reward',
    version: 'v1',
    boardType,
    description: 'Player 1 chooses minimum collapse for line of 5.',
    state: toFixtureState(state as EngineGameState, boardType),
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateChooseLineRewardCollapseAllFixture(
  engineState: EngineGameState
): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 has a line of 5 markers (allowing choice).
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'line_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  const line: LineInfo = {
    positions: [
      { x: 0, y: 0 },
      { x: 0, y: 1 },
      { x: 0, y: 2 },
      { x: 0, y: 3 },
      { x: 0, y: 4 },
    ],
    player: 1,
    length: 5,
    direction: { x: 0, y: 1 },
  };

  state.board.formedLines = [line];

  const sharedBefore = toFixtureState(state as EngineGameState, boardType);

  // Option 1: Collapse all markers in the line.
  const collapsedMarkers = [...line.positions];

  const move: Move = {
    id: 'fixture-square8-2p-choose-line-reward-collapse-all',
    type: 'choose_line_reward',
    player: 1,
    to: { x: 0, y: 0 }, // Sentinel
    formedLines: [line],
    collapsedMarkers,
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ChooseLineRewardAction = {
    type: 'CHOOSE_LINE_REWARD',
    playerId: 1,
    lineIndex: 0,
    selection: 'COLLAPSE_ALL',
    collapsedPositions: collapsedMarkers,
  };

  const tsValidation = validateChooseLineReward(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.choose_line_reward.collapse_all',
      version: 'v1',
      boardType,
      description: 'Player 1 chooses collapse-all reward for line of 5.',
      state: sharedBefore,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.choose_line_reward.collapse_all',
    version: 'v1',
    boardType,
    description: 'Player 1 chooses collapse-all reward for line of 5.',
    state: sharedBefore,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateProcessTerritoryFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 has a disconnected territory region.
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'territory_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  const regionId = 'disconnected-0';
  const region: Territory = {
    spaces: [{ x: 0, y: 0 }],
    controllingPlayer: 1,
    isDisconnected: true,
  };

  state.board.territories.set(regionId, region);

  // Need a stack outside to pay cost
  state.board.stacks.set('7,7', {
    position: { x: 7, y: 7 },
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 1,
  });

  const fixtureState = toFixtureState(state, boardType);

  const move: Move = {
    id: 'fixture-square8-2p-process-territory',
    type: 'process_territory_region',
    player: 1,
    to: { x: 0, y: 0 }, // Sentinel
    disconnectedRegions: [region],
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ProcessTerritoryAction = {
    type: 'PROCESS_TERRITORY',
    playerId: 1,
    regionId,
  };

  const tsValidation = validateProcessTerritory(state, action);

  // Deep clone state for engine execution to avoid side-effects on fixtureState
  // (TerritoryMutator currently mutates territory objects in place)

  // Instead, let's just clone the territory object in the state passed to engine.
  const stateForEngine = {
    ...state,
    board: {
      ...state.board,
      territories: new Map(),
    },
  };
  // eslint-disable-next-line no-restricted-syntax
  for (const [k, v] of state.board.territories) {
    stateForEngine.board.territories.set(k, { ...v });
  }

  const engine = new GameEngine(stateForEngine);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.process_territory',
      version: 'v1',
      boardType,
      description: 'Player 1 processes disconnected territory.',
      state: fixtureState,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.process_territory',
    version: 'v1',
    boardType,
    description: 'Player 1 processes disconnected territory.',
    state: fixtureState,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateEliminateStackFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 needs to eliminate rings (e.g. after territory processing).
  // ELIMINATE_STACK is the explicit self-elimination step that typically
  // follows PROCESS_TERRITORY when a player must pay a cost from an
  // off-region stack. Here we construct a minimal structurally-valid
  // scenario that exercises the canonical validator.

  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'territory_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  // Stack to eliminate from
  const stackPos = { x: 7, y: 7 };
  state.board.stacks.set('7,7', {
    position: stackPos,
    rings: [1, 1],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: 1,
  });

  const fixtureState = toFixtureState(state, boardType);

  const move: Move = {
    id: 'fixture-square8-2p-eliminate-stack',
    type: 'eliminate_rings_from_stack',
    player: 1,
    to: stackPos,
    eliminatedRings: [{ player: 1, count: 1 }],
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: EliminateStackAction = {
    type: 'ELIMINATE_STACK',
    playerId: 1,
    stackPosition: stackPos,
  };

  const tsValidation = validateEliminateStack(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.eliminate_stack',
      version: 'v1',
      boardType,
      description: 'Player 1 eliminates ring from stack.',
      state: fixtureState,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.eliminate_stack',
    version: 'v1',
    boardType,
    description: 'Player 1 eliminates ring from stack.',
    state: fixtureState,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateSkipPlacementFixture(engineState: EngineGameState): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 in placement phase, with at least one legal move from a
  // controlled stack, so placement is optional and skip_placement should be
  // considered a legal no-op that advances to movement.
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'ring_placement';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  // Add a single controllable stack in open space so that at least one
  // legal move or capture is available according to movement rules.
  state.board.stacks.set('3,3', {
    position: { x: 3, y: 3 },
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 1,
  });

  const fixtureState = toFixtureState(state, boardType);

  const move: Move = {
    id: 'fixture-square8-2p-skip-placement',
    type: 'skip_placement',
    player: 1,
    to: { x: 0, y: 0 }, // Sentinel
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: SkipPlacementAction = {
    type: 'SKIP_PLACEMENT',
    playerId: 1,
  };

  const tsValidation = validateSkipPlacement(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: 'square8_2p.skip_placement',
      version: 'v1',
      boardType,
      description: 'Player 1 skips placement (optional placement case).',
      state: fixtureState,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: 'square8_2p.skip_placement',
    version: 'v1',
    boardType,
    description: 'Player 1 skips placement (optional placement case).',
    state: fixtureState,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateSkipPlacementNoStacksInvalidFixture(
  engineState: EngineGameState
): StateActionFixture {
  const boardType: BoardType = 'square8';

  // Setup: Player 1 in placement phase with rings in hand but NO controlled
  // stacks on the board. Per validateSkipPlacement, this should be illegal
  // (placement is mandatory / leads to forced elimination), so tsValid=false
  // and no tsNext snapshot.
  const state = JSON.parse(JSON.stringify(engineState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'ring_placement';

  // Clear the board: no stacks, markers, or territories.
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  const fixtureState = toFixtureState(state, boardType);

  const move: Move = {
    id: 'fixture-square8-2p-skip-placement-no-stacks',
    type: 'skip_placement',
    player: 1,
    to: { x: 0, y: 0 }, // Sentinel
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: SkipPlacementAction = {
    type: 'SKIP_PLACEMENT',
    playerId: 1,
  };

  const tsValidation = validateSkipPlacement(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type === 'ACTION_PROCESSED') {
    // This would indicate a mismatch between validator and engine wiring;
    // still emit tsNext for diagnostic purposes.
    const engineAfter = engine.getGameState();
    const sharedAfter = toSharedState(engineAfter, boardType);
    return {
      id: 'square8_2p.skip_placement.no_stacks',
      version: 'v1',
      boardType,
      description:
        'Player 1 attempts to skip placement with no controlled stacks (should be invalid).',
      state: fixtureState,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
        tsNext: snapshotState(sharedAfter),
      },
    };
  }

  return {
    id: 'square8_2p.skip_placement.no_stacks',
    version: 'v1',
    boardType,
    description:
      'Player 1 attempts to skip placement with no controlled stacks (should be invalid).',
    state: fixtureState,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
    },
  };
}

function generateMultiLineFixtureForBoard(
  baseState: EngineGameState,
  boardType: BoardType,
  version: FixtureVersion
): StateActionFixture {
  const state = JSON.parse(JSON.stringify(baseState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'line_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  let lines: LineInfo[];
  if (boardType === 'square8') {
    lines = [
      {
        positions: [
          { x: 1, y: 1 },
          { x: 2, y: 1 },
          { x: 3, y: 1 },
          { x: 4, y: 1 },
        ],
        player: 1,
        length: 4,
        direction: { x: 1, y: 0 },
      },
      {
        positions: [
          { x: 2, y: 0 },
          { x: 2, y: 1 },
          { x: 2, y: 2 },
          { x: 2, y: 3 },
        ],
        player: 1,
        length: 4,
        direction: { x: 0, y: 1 },
      },
      {
        positions: [
          { x: 4, y: 4 },
          { x: 5, y: 5 },
          { x: 6, y: 6 },
          { x: 7, y: 7 },
        ],
        player: 2,
        length: 4,
        direction: { x: 1, y: 1 },
      },
    ];
  } else if (boardType === 'square19') {
    lines = [
      {
        positions: [
          { x: 8, y: 9 },
          { x: 9, y: 9 },
          { x: 10, y: 9 },
          { x: 11, y: 9 },
        ],
        player: 1,
        length: 4,
        direction: { x: 1, y: 0 },
      },
      {
        positions: [
          { x: 9, y: 8 },
          { x: 9, y: 9 },
          { x: 9, y: 10 },
          { x: 9, y: 11 },
        ],
        player: 1,
        length: 4,
        direction: { x: 0, y: 1 },
      },
      {
        positions: [
          { x: 5, y: 5 },
          { x: 6, y: 6 },
          { x: 7, y: 7 },
          { x: 8, y: 8 },
        ],
        player: 2,
        length: 4,
        direction: { x: 1, y: 1 },
      },
    ];
  } else {
    // hexagonal board: use axial-like coordinates with z for clarity.
    lines = [
      {
        positions: [
          { x: 0, y: 0, z: 0 },
          { x: 1, y: -1, z: 0 },
          { x: 2, y: -2, z: 0 },
          { x: 3, y: -3, z: 0 },
        ],
        player: 1,
        length: 4,
        direction: { x: 1, y: -1 },
      },
      {
        positions: [
          { x: 0, y: 1, z: -1 },
          { x: 1, y: 0, z: -1 },
          { x: 2, y: -1, z: -1 },
          { x: 3, y: -2, z: -1 },
        ],
        player: 1,
        length: 4,
        direction: { x: 1, y: -1 },
      },
      {
        positions: [
          { x: -1, y: 1, z: 0 },
          { x: -2, y: 2, z: 0 },
          { x: -3, y: 3, z: 0 },
          { x: -4, y: 4, z: 0 },
        ],
        player: 2,
        length: 4,
        direction: { x: -1, y: 1 },
      },
    ];
  }

  state.board.formedLines = lines;

  const fixtureState = toFixtureState(state as EngineGameState, boardType);

  // For the unified Move model and Python mirror, the canonical way to
  // identify the line being processed is via move.formedLines[0] and its
  // first position. The TS engine still uses lineIndex internally, but for
  // parity we ensure that the Move we emit carries only the selected line
  // (the second line in `lines`) so that Python can resolve the same target
  // without needing an explicit index.
  const targetLine = lines[1];

  const move: Move = {
    id: `fixture-${boardType}-2p-process-line-multiple`,
    type: 'process_line',
    player: 1,
    to: targetLine.positions[0],
    formedLines: [targetLine],
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ProcessLineAction = {
    type: 'PROCESS_LINE',
    playerId: 1,
    lineIndex: 1, // process the second line to exercise lineIndex semantics
  };

  const tsValidation = validateProcessLine(state, action);

  const engine = new GameEngine(state);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: `${boardType}_2p.process_line.multiple_lines`,
      version,
      boardType,
      description: `Multi-line line_processing scenario on ${boardType} board with overlapping and mixed-player lines.`,
      state: fixtureState,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: `${boardType}_2p.process_line.multiple_lines`,
    version,
    boardType,
    description: `Multi-line line_processing scenario on ${boardType} board with overlapping and mixed-player lines.`,
    state: fixtureState,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function generateMultiRegionTerritoryFixtureForBoard(
  baseState: EngineGameState,
  boardType: BoardType,
  version: FixtureVersion
): StateActionFixture {
  const state = JSON.parse(JSON.stringify(baseState)) as any;
  state.currentPlayer = 1;
  state.currentPhase = 'territory_processing';
  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();

  // Three regions: two for player 1 (one disconnected) and one for player 2.
  const region1: Territory = {
    spaces: [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
    ],
    controllingPlayer: 1,
    isDisconnected: false,
  };

  const region2: Territory = {
    spaces: [
      { x: 3, y: 3 },
      { x: 3, y: 4 },
    ],
    controllingPlayer: 1,
    isDisconnected: true,
  };

  const region3: Territory = {
    spaces: [
      { x: -2, y: -2 },
      { x: -1, y: -2 },
    ],
    controllingPlayer: 2,
    isDisconnected: true,
  };

  state.board.territories.set('region-1', region1);
  state.board.territories.set('region-2', region2);
  state.board.territories.set('region-3', region3);

  // Off-region stacks for both players so that PROCESS_TERRITORY has a
  // realistic cost-paying context.
  state.board.stacks.set('7,7', {
    position: { x: 7, y: 7 },
    rings: [1, 1],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: 1,
  });

  state.board.stacks.set('6,6', {
    position: { x: 6, y: 6 },
    rings: [2],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 2,
  });

  const fixtureState = toFixtureState(state as EngineGameState, boardType);

  const move: Move = {
    id: `fixture-${boardType}-2p-process-territory-multi-region`,
    type: 'process_territory_region',
    player: 1,
    to: { x: 0, y: 0 },
    disconnectedRegions: [region2],
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: state.moveHistory.length + 1,
  };

  const action: ProcessTerritoryAction = {
    type: 'PROCESS_TERRITORY',
    playerId: 1,
    regionId: 'region-2',
  };

  const tsValidation = validateProcessTerritory(state, action);

  // Clone territories into a fresh Map for the engine to avoid mutating
  // the fixture state, mirroring the v1 territory fixture pattern.
  const stateForEngine = {
    ...state,
    board: {
      ...state.board,
      territories: new Map<string, Territory>(),
    },
  };
  // eslint-disable-next-line no-restricted-syntax
  for (const [k, v] of state.board.territories as Map<string, Territory>) {
    stateForEngine.board.territories.set(k, { ...v });
  }

  const engine = new GameEngine(stateForEngine as EngineGameState);
  const event = engine.processAction(action);

  if (event.type !== 'ACTION_PROCESSED') {
    return {
      id: `${boardType}_2p.process_territory.multi_region`,
      version,
      boardType,
      description: `Multi-region territory_processing scenario on ${boardType} board with multiple disconnected regions.`,
      state: fixtureState,
      move,
      expected: {
        tsValid: tsValidation.valid,
        tsValidation,
      },
    };
  }

  const engineAfter = engine.getGameState();
  const sharedAfter = toSharedState(engineAfter, boardType);

  return {
    id: `${boardType}_2p.process_territory.multi_region`,
    version,
    boardType,
    description: `Multi-region territory_processing scenario on ${boardType} board with multiple disconnected regions.`,
    state: fixtureState,
    move,
    expected: {
      tsValid: tsValidation.valid,
      tsValidation,
      tsNext: snapshotState(sharedAfter),
    },
  };
}

function writeFixtures(outDir: string, fixtures: { name: string; data: unknown }[]): void {
  mkdirSync(outDir, { recursive: true });

  for (const fixture of fixtures) {
    writeFileSync(join(outDir, fixture.name), JSON.stringify(fixture.data, null, 2), 'utf-8');
  }

  // eslint-disable-next-line no-console
  console.log(`Wrote rules parity fixtures to ${outDir}`);
}

function generateV1Fixtures(): void {
  const outDirV1 = join(__dirname, '..', 'fixtures', 'rules-parity', 'v1');

  const players = makePlayers(2);
  const timeControl = makeTimeControl();
  const boardType: BoardType = 'square8';

  // Base engine state
  const initialEngineState = createInitialGameState(
    'rules-parity-square8-2p',
    boardType,
    players,
    timeControl,
    false
  );

  const initialSquare8 = generateSquare8TwoPlayerInitial(initialEngineState);

  // Placement fixture and state after placement
  const { fixture: placementFixture, engineAfter: engineAfterPlacement } =
    generatePlacementFixture(initialEngineState);

  // Movement fixture from the post-placement state
  const moveStackFixture = generateMoveStackFixture(engineAfterPlacement);

  // Additional v1 fixtures
  const overtakingCaptureFixture = generateOvertakingCaptureFixture(initialEngineState);
  const continueCaptureFixture = generateContinueCaptureFixture(initialEngineState);
  const processLineFixture = generateProcessLineFixture(initialEngineState);
  const chooseLineRewardFixture = generateChooseLineRewardFixture(initialEngineState);
  const chooseLineRewardCollapseAllFixture =
    generateChooseLineRewardCollapseAllFixture(initialEngineState);
  const processTerritoryFixture = generateProcessTerritoryFixture(initialEngineState);
  const eliminateStackFixture = generateEliminateStackFixture(initialEngineState);
  const skipPlacementFixture = generateSkipPlacementFixture(initialEngineState);
  const skipPlacementNoStacksFixture =
    generateSkipPlacementNoStacksInvalidFixture(initialEngineState);

  // Multi-step trace fixture: placement followed by movement.
  const placementAndMovementTrace: TraceFixture = {
    version: 'v1',
    boardType,
    initialState: initialSquare8.state as SharedGameState,
    steps: [
      {
        label: 'place_ring_center',
        move: placementFixture.move,
        expected: {
          tsValid: placementFixture.expected.tsValid,
          tsStateHash: placementFixture.expected.tsNext?.stateHash,
          tsS: placementFixture.expected.tsNext?.S,
        },
        stateHash: placementFixture.expected.tsNext?.stateHash,
        sInvariant: placementFixture.expected.tsNext?.S,
      },
      {
        label: 'move_stack_forward',
        move: moveStackFixture.move,
        expected: {
          tsValid: moveStackFixture.expected.tsValid,
          tsStateHash: moveStackFixture.expected.tsNext?.stateHash,
          tsS: moveStackFixture.expected.tsNext?.S,
        },
        stateHash: moveStackFixture.expected.tsNext?.stateHash,
        sInvariant: moveStackFixture.expected.tsNext?.S,
      },
    ],
  };

  // Single-step traces derived from state+action fixtures.
  const overtakingCaptureTrace = makeTraceFromStateAction(
    boardType,
    overtakingCaptureFixture,
    'overtaking_capture'
  );
  const continueCaptureTrace = makeTraceFromStateAction(
    boardType,
    continueCaptureFixture,
    'continue_capture'
  );
  const processLineTrace = makeTraceFromStateAction(boardType, processLineFixture, 'process_line');
  const chooseLineRewardMinimumTrace = makeTraceFromStateAction(
    boardType,
    chooseLineRewardFixture,
    'choose_line_reward_minimum_collapse'
  );
  const chooseLineRewardCollapseAllTrace = makeTraceFromStateAction(
    boardType,
    chooseLineRewardCollapseAllFixture,
    'choose_line_reward_collapse_all'
  );
  const processTerritoryTrace = makeTraceFromStateAction(
    boardType,
    processTerritoryFixture,
    'process_territory_region'
  );
  const eliminateStackTrace = makeTraceFromStateAction(
    boardType,
    eliminateStackFixture,
    'eliminate_stack'
  );
  const skipPlacementTrace = makeTraceFromStateAction(
    boardType,
    skipPlacementFixture,
    'skip_placement'
  );

  const fixtures = [
    { name: 'state_only.square8_2p.initial.json', data: initialSquare8 },
    {
      name: 'state_action.square8_2p.place_ring_center.json',
      data: placementFixture,
    },
    {
      name: 'state_action.square8_2p.move_stack_forward.json',
      data: moveStackFixture,
    },
    {
      name: 'state_action.square8_2p.overtaking_capture.json',
      data: overtakingCaptureFixture,
    },
    {
      name: 'state_action.square8_2p.continue_capture.json',
      data: continueCaptureFixture,
    },
    {
      name: 'state_action.square8_2p.process_line.json',
      data: processLineFixture,
    },
    {
      name: 'state_action.square8_2p.choose_line_reward.json',
      data: chooseLineRewardFixture,
    },
    {
      name: 'state_action.square8_2p.choose_line_reward.collapse_all.json',
      data: chooseLineRewardCollapseAllFixture,
    },
    {
      name: 'state_action.square8_2p.process_territory.json',
      data: processTerritoryFixture,
    },
    {
      name: 'state_action.square8_2p.eliminate_stack.json',
      data: eliminateStackFixture,
    },
    {
      name: 'state_action.square8_2p.skip_placement.json',
      data: skipPlacementFixture,
    },
    {
      name: 'state_action.square8_2p.skip_placement.no_stacks.json',
      data: skipPlacementNoStacksFixture,
    },
    {
      name: 'trace.square8_2p.placement_and_movement.json',
      data: placementAndMovementTrace,
    },
    {
      name: 'trace.square8_2p.overtaking_capture.json',
      data: overtakingCaptureTrace,
    },
    {
      name: 'trace.square8_2p.continue_capture.json',
      data: continueCaptureTrace,
    },
    {
      name: 'trace.square8_2p.process_line.json',
      data: processLineTrace,
    },
    {
      name: 'trace.square8_2p.choose_line_reward.minimum_collapse.json',
      data: chooseLineRewardMinimumTrace,
    },
    {
      name: 'trace.square8_2p.choose_line_reward.collapse_all.json',
      data: chooseLineRewardCollapseAllTrace,
    },
    {
      name: 'trace.square8_2p.process_territory.json',
      data: processTerritoryTrace,
    },
    {
      name: 'trace.square8_2p.eliminate_stack.json',
      data: eliminateStackTrace,
    },
    {
      name: 'trace.square8_2p.skip_placement.json',
      data: skipPlacementTrace,
    },
  ];

  writeFixtures(outDirV1, fixtures);
}

function generateV2Fixtures(): void {
  const outDirV2 = join(__dirname, '..', 'fixtures', 'rules-parity', 'v2');

  const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal'];

  const fixtures: { name: string; data: unknown }[] = [];

  for (const boardType of boardTypes) {
    const players = makePlayers(2);
    const timeControl = makeTimeControl();

    const baseState = createInitialGameState(
      `rules-parity-${boardType}-2p-v2`,
      boardType,
      players,
      timeControl,
      false
    );

    const multiLineFixture = generateMultiLineFixtureForBoard(baseState, boardType, 'v2');
    const multiRegionFixture = generateMultiRegionTerritoryFixtureForBoard(
      baseState,
      boardType,
      'v2'
    );

    const multiLineTrace = makeTraceFromStateAction(
      boardType,
      multiLineFixture,
      'process_line.multiple_lines'
    );
    const multiRegionTrace = makeTraceFromStateAction(
      boardType,
      multiRegionFixture,
      'process_territory.multi_region'
    );

    const prefix = `${boardType}_2p`;

    fixtures.push(
      {
        name: `state_action.${prefix}.process_line.multiple_lines.json`,
        data: multiLineFixture,
      },
      {
        name: `state_action.${prefix}.process_territory.multi_region.json`,
        data: multiRegionFixture,
      },
      {
        name: `trace.${prefix}.process_line.multiple_lines.json`,
        data: multiLineTrace,
      },
      {
        name: `trace.${prefix}.process_territory.multi_region.json`,
        data: multiRegionTrace,
      }
    );
  }

  if (fixtures.length > 0) {
    writeFixtures(outDirV2, fixtures);
  }
}

function main() {
  generateV1Fixtures();
  generateV2Fixtures();
}

main();
