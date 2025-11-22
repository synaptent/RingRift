import { GameEngine as SharedGameEngine } from '../../src/shared/engine/GameEngine';
import {
  GameState as SharedEngineGameState,
  PlaceRingAction,
  MoveStackAction,
} from '../../src/shared/engine/types';
import { GameEngine as LegacyGameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  Territory,
  positionToString,
} from '../../src/shared/types/game';
import {
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../src/shared/engine/core';
import {
  moveToGameAction,
  gameActionToMove,
  MoveMappingError,
  BareMove,
} from '../../src/shared/engine/moveActionAdapter';
import { createInitialGameState } from '../../src/shared/engine/initialState';

/**
 * Helpers shared across adapter tests
 */

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'ai',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600 * 1000,
      ringsInHand: 30,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'ai',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600 * 1000,
      ringsInHand: 30,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

const timeControl: any = { initialTime: 600, increment: 0, type: 'blitz' };

function createEngines(gameId: string, boardType: BoardType = 'square8') {
  const players = createPlayers();
  const legacy = new LegacyGameEngine(gameId, boardType, players, timeControl, true);
  legacy.startGame();

  const baseState = legacy.getGameState();
  const shared = new SharedGameEngine(baseState as unknown as SharedEngineGameState);

  return { legacy, shared };
}

function paritySnapshot(state: GameState) {
  const summary = summarizeBoard(state.board);
  const progress = computeProgressSnapshot(state);

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
    S: summary.markers.length + summary.collapsedSpaces.length + progress.eliminated,
    stateHash: hashGameState(state),
    stacks: Array.from(state.board.stacks.entries()),
    markersByKey: Array.from(state.board.markers.entries()),
    collapsedByKey: Array.from(state.board.collapsedSpaces.entries()),
  };
}

function stripBareMove(move: BareMove): Record<string, unknown> {
  // Normalise by dropping non-semantic/transient fields if any ever appear.
  const {
    // keep: type, player, from, to, captureTarget, formedLines, disconnectedRegions,
    // eliminatedRings, eliminationFromStack, placementCount, placedOnStack, captureType,
    // etc.
    // Drop nothing for now except thinkTime which is always 0 in adapter output.
    thinkTime, // eslint-disable-line @typescript-eslint/no-unused-vars
    ...rest
  } = move as any;
  return rest;
}

/**
 * Unit-level adapter round-trip tests (Move <-> GameAction)
 */
describe('Move↔GameAction adapter – round-trip semantics', () => {
  it('maps place_ring on empty cell to PLACE_RING and back', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-place-empty',
      'square8',
      players,
      timeControl
    );

    const move: Move = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      placementCount: 2,
      placedOnStack: false,
      timestamp: new Date(),
      thinkTime: 123,
      moveNumber: 1,
    };

    const action = moveToGameAction(move, sharedState);
    expect(action).toEqual<PlaceRingAction>({
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 2,
    });

    const back = gameActionToMove(action, sharedState);
    expect(stripBareMove(back)).toEqual({
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      placementCount: 2,
      placedOnStack: false,
    });
  });

  it('maps place_ring onto an existing stack with placedOnStack hint', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-place-stack',
      'square8',
      players,
      timeControl
    );

    // Seed an existing stack at (0,0) so actionToPlaceRingMove can infer placedOnStack=true.
    const key = '0,0';
    (sharedState.board.stacks as any).set(key, {
      position: { x: 0, y: 0 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    });

    const move: Move = {
      id: 'm2',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      placementCount: 1,
      placedOnStack: true,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const action = moveToGameAction(move, sharedState);
    expect(action).toEqual<PlaceRingAction>({
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 1,
    });

    const back = gameActionToMove(action, sharedState);
    expect(stripBareMove(back)).toEqual({
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      placementCount: 1,
      placedOnStack: true,
    });
  });

  it('maps skip_placement to SKIP_PLACEMENT and back with sentinel position', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-skip-placement',
      'square8',
      players,
      timeControl
    );

    const move: Move = {
      id: 'm3',
      type: 'skip_placement',
      player: 1,
      to: { x: 42, y: 99 }, // arbitrary; ignored semantically
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const action = moveToGameAction(move, sharedState);
    expect(action).toEqual({
      type: 'SKIP_PLACEMENT',
      playerId: 1,
    });

    const back = gameActionToMove(action, sharedState);
    expect(back.type).toBe('skip_placement');
    expect(back.player).toBe(1);
    // Adapter uses a fixed sentinel to for `to`; we only care that it exists.
    expect(back.to).toBeDefined();
  });

  it('maps move_stack to MOVE_STACK and back', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-move-stack',
      'square8',
      players,
      timeControl
    );

    const move: Move = {
      id: 'm4',
      type: 'move_stack',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 3 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const action = moveToGameAction(move, sharedState);
    expect(action).toEqual<MoveStackAction>({
      type: 'MOVE_STACK',
      playerId: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 3 },
    });

    const back = gameActionToMove(action, sharedState);
    expect(stripBareMove(back)).toEqual({
      type: 'move_stack',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 3 },
    });
  });

  it('maps overtaking_capture and continue_capture_segment to capture actions and back', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState('adapter-capture', 'square8', players, timeControl);

    const base: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 0, y: 2 },
      to: { x: 0, y: 4 },
      thinkTime: 0,
    };

    const overMove: Move = {
      ...base,
      id: 'cap-1',
      timestamp: new Date(),
      moveNumber: 1,
    };

    const overAction = moveToGameAction(overMove, sharedState);
    expect(overAction.type).toBe('OVERTAKING_CAPTURE');
    expect((overAction as any).playerId).toBe(1);
    expect((overAction as any).from).toEqual({ x: 0, y: 0 });
    expect((overAction as any).captureTarget).toEqual({ x: 0, y: 2 });
    expect((overAction as any).to).toEqual({ x: 0, y: 4 });

    const overBack = gameActionToMove(overAction, sharedState);
    expect(stripBareMove(overBack)).toEqual({
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 0, y: 2 },
      to: { x: 0, y: 4 },
    });

    const contMove: Move = {
      ...base,
      type: 'continue_capture_segment',
      id: 'cap-2',
      timestamp: new Date(),
      moveNumber: 2,
    };
    const contAction = moveToGameAction(contMove, sharedState);
    expect(contAction.type).toBe('CONTINUE_CHAIN');

    const contBack = gameActionToMove(contAction, sharedState);
    expect(stripBareMove(contBack)).toEqual({
      type: 'continue_capture_segment',
      player: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 0, y: 2 },
      to: { x: 0, y: 4 },
    });
  });

  it('maps process_line / choose_line_reward using formedLines metadata and back', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState('adapter-lines', 'square8', players, timeControl);

    const linePositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
    ];

    const line: any = {
      player: 1,
      positions: linePositions,
      length: linePositions.length,
      direction: { x: 1, y: 0 },
    };

    (sharedState.board as any).formedLines = [line];

    const processMove: Move = {
      id: 'process-line-0',
      type: 'process_line',
      player: 1,
      to: linePositions[0],
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const processAction = moveToGameAction(processMove, sharedState);
    expect(processAction).toEqual({
      type: 'PROCESS_LINE',
      playerId: 1,
      lineIndex: 0,
    });

    const processBack = gameActionToMove(processAction, sharedState);
    expect(processBack.type).toBe('process_line');
    expect(processBack.player).toBe(1);
    expect(processBack.formedLines).toBeDefined();
    expect(processBack.formedLines![0].positions).toEqual(linePositions);

    // choose_line_reward with minimum-collapse subset
    const minSubset = linePositions.slice(0, 3);
    const chooseMove: Move = {
      id: 'choose-line-0',
      type: 'choose_line_reward',
      player: 1,
      to: linePositions[0],
      formedLines: [line],
      collapsedMarkers: minSubset,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 2,
    };

    const chooseAction = moveToGameAction(chooseMove, sharedState);
    expect(chooseAction.type).toBe('CHOOSE_LINE_REWARD');
    expect((chooseAction as any).selection).toBe('MINIMUM_COLLAPSE');
    expect((chooseAction as any).collapsedPositions).toEqual(minSubset);

    const chooseBack = gameActionToMove(chooseAction, sharedState);
    expect(chooseBack.type).toBe('choose_line_reward');
    expect(chooseBack.player).toBe(1);
    expect((chooseBack as any).collapsedMarkers).toEqual(minSubset);
  });

  it('maps process_territory_region using disconnectedRegions metadata and back', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-territory',
      'square8',
      players,
      timeControl
    );

    const regionSpaces: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
    ];

    const region: Territory = {
      spaces: regionSpaces,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    (sharedState.board.territories as any).set('region-1', region);

    const move: Move = {
      id: 'proc-region',
      type: 'process_territory_region',
      player: 1,
      to: regionSpaces[0],
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const action = moveToGameAction(move, sharedState);
    expect(action).toEqual({
      type: 'PROCESS_TERRITORY',
      playerId: 1,
      regionId: 'region-1',
    });

    const back = gameActionToMove(action, sharedState);
    expect(back.type).toBe('process_territory_region');
    expect(back.player).toBe(1);
    expect(back.disconnectedRegions).toBeDefined();
    expect(back.disconnectedRegions![0].spaces).toEqual(regionSpaces);
  });

  it('maps eliminate_rings_from_stack to ELIMINATE_STACK and back with diagnostic fields', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-eliminate',
      'square8',
      players,
      timeControl
    );

    const stackPos: Position = { x: 0, y: 0 };
    const stackKey = positionToString(stackPos);
    (sharedState.board.stacks as any).set(stackKey, {
      position: stackPos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    });

    const move: Move = {
      id: 'elim-1',
      type: 'eliminate_rings_from_stack',
      player: 1,
      to: stackPos,
      eliminatedRings: [{ player: 1, count: 2 }],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const action = moveToGameAction(move, sharedState);
    expect(action).toEqual({
      type: 'ELIMINATE_STACK',
      playerId: 1,
      stackPosition: stackPos,
    });

    const back = gameActionToMove(action, sharedState);
    expect(back.type).toBe('eliminate_rings_from_stack');
    expect(back.player).toBe(1);
    expect(back.to).toEqual(stackPos);
    expect((back as any).eliminationFromStack).toBeDefined();
    expect((back as any).eliminationFromStack.position).toEqual(stackPos);
  });

  it('throws MoveMappingError for unsupported legacy/experimental move types', () => {
    const players = createPlayers();
    const sharedState = createInitialGameState(
      'adapter-unsupported',
      'square8',
      players,
      timeControl
    );

    const legacyTypes: Move['type'][] = ['build_stack', 'line_formation', 'territory_claim'];

    for (const t of legacyTypes) {
      const move: Move = {
        id: `legacy-${t}`,
        type: t as any,
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      expect(() => moveToGameAction(move, sharedState)).toThrow(MoveMappingError);
    }
  });
});

/**
 * Integration tests: drive a small scenario through both:
 * - Legacy backend GameEngine.makeMove using Move
 * - Shared GameEngine.processAction using GameAction derived via the adapter
 * and assert state / S-invariant equality.
 */
describe('Move↔GameAction adapter – integration parity with legacy GameEngine', () => {
  it('single place_ring move keeps legacy and shared states in lockstep', async () => {
    const { legacy, shared } = createEngines('adapter-parity-place');

    const legacyState0 = legacy.getGameState();
    const sharedState0 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState0)).toEqual(paritySnapshot(legacyState0));

    // Step 1: player 1 places a single ring at (0,0).
    const placePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      thinkTime: 0,
    };

    const legacyResult = await legacy.makeMove(placePayload);
    expect(legacyResult.success).toBe(true);

    const sharedBeforeForAdapter = shared.getGameState() as unknown as SharedEngineGameState;
    const placeForAdapter: Move = {
      ...placePayload,
      id: 'adapter-place-1',
      timestamp: new Date(),
      moveNumber: sharedBeforeForAdapter.moveHistory.length + 1,
    };

    const placeAction = moveToGameAction(placeForAdapter, sharedBeforeForAdapter);
    const sharedEvent = shared.processAction(placeAction);
    expect(sharedEvent.type).toBe('ACTION_PROCESSED');

    const legacyState1 = legacy.getGameState();
    const sharedState1 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState1)).toEqual(paritySnapshot(legacyState1));
  });

  it('place_ring + move_stack sequence via adapter keeps states and invariants aligned', async () => {
    const { legacy, shared } = createEngines('adapter-parity-sequence');

    const legacyState0 = legacy.getGameState();
    const sharedState0 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState0)).toEqual(paritySnapshot(legacyState0));

    // Step 1: player 1 places at (0,0).
    const placePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      thinkTime: 0,
    };

    const legacyPlaceResult = await legacy.makeMove(placePayload);
    expect(legacyPlaceResult.success).toBe(true);

    const sharedBeforePlace = shared.getGameState() as unknown as SharedEngineGameState;
    const placeMoveForAdapter: Move = {
      ...placePayload,
      id: 'adapter-seq-place',
      timestamp: new Date(),
      moveNumber: sharedBeforePlace.moveHistory.length + 1,
    };
    const placeAction = moveToGameAction(placeMoveForAdapter, sharedBeforePlace);
    const sharedPlaceEvent = shared.processAction(placeAction);
    expect(sharedPlaceEvent.type).toBe('ACTION_PROCESSED');

    const legacyState1 = legacy.getGameState();
    const sharedState1 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState1)).toEqual(paritySnapshot(legacyState1));

    // Step 2: same player moves stack from (0,0) to (0,1).
    const legacyStateForMove = legacy.getGameState();
    const movePlayer = legacyStateForMove.currentPlayer;

    const movePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'move_stack',
      player: movePlayer,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
      thinkTime: 0,
    };

    const legacyMoveResult = await legacy.makeMove(movePayload);
    expect(legacyMoveResult.success).toBe(true);

    const sharedBeforeMove = shared.getGameState() as unknown as SharedEngineGameState;
    const moveForAdapter: Move = {
      ...movePayload,
      id: 'adapter-seq-move',
      timestamp: new Date(),
      moveNumber: sharedBeforeMove.moveHistory.length + 1,
    };
    const moveAction = moveToGameAction(moveForAdapter, sharedBeforeMove);
    const sharedMoveEvent = shared.processAction(moveAction);
    expect(sharedMoveEvent.type).toBe('ACTION_PROCESSED');

    const legacyState2 = legacy.getGameState();
    const sharedState2 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState2)).toEqual(paritySnapshot(legacyState2));
  });
});
describe('Move↔GameAction adapter – capture parity scenario', () => {
  it('overtaking capture via Move + adapter yields aligned states', async () => {
    const players = createPlayers();
    const boardType: BoardType = 'square8';

    const legacy = new LegacyGameEngine(
      'adapter-parity-capture',
      boardType,
      players,
      timeControl,
      true
    );
    legacy.startGame();

    // Build a simple overtaking capture scenario via legacy makeMove:
    // 1. P1 places at (0,0) and moves to (0,1).
    // 2. P2 places at (0,3) and moves to (0,4).
    // 3. P1 places at (0,2).
    const setupMoves: Array<Omit<Move, 'id' | 'timestamp' | 'moveNumber'>> = [
      {
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        thinkTime: 0,
      },
      {
        type: 'move_stack',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 1 },
        thinkTime: 0,
      },
      {
        type: 'place_ring',
        player: 2,
        to: { x: 0, y: 3 },
        thinkTime: 0,
      },
      {
        type: 'move_stack',
        player: 2,
        from: { x: 0, y: 3 },
        to: { x: 0, y: 4 },
        thinkTime: 0,
      },
      {
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 2 },
        placementCount: 3,
        thinkTime: 0,
      },
    ];

    for (const payload of setupMoves) {
      const res = await legacy.makeMove(payload);
      expect(res.success).toBe(true);
    }

    // Recreate the shared engine from the pre-capture legacy state so both
    // engines see an identical board and player/phase metadata.
    const preCaptureLegacy = legacy.getGameState();
    const shared = new SharedGameEngine(preCaptureLegacy as unknown as SharedEngineGameState);

    // Overtaking capture: P1 from (0,2) over (0,4) landing at (0,5).
    const capturePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'overtaking_capture',
      player: preCaptureLegacy.currentPlayer,
      from: { x: 0, y: 2 },
      captureTarget: { x: 0, y: 4 },
      to: { x: 0, y: 5 },
      thinkTime: 0,
    };

    const legacyCaptureRes = await legacy.makeMove(capturePayload);
    expect(legacyCaptureRes.success).toBe(true);

    const sharedBefore = shared.getGameState() as unknown as SharedEngineGameState;
    const fullCaptureMove: Move = {
      ...capturePayload,
      id: 'adapter-parity-capture-move',
      timestamp: new Date(),
      moveNumber: sharedBefore.moveHistory.length + 1,
    };

    const captureAction = moveToGameAction(fullCaptureMove, sharedBefore);
    const sharedEvent = shared.processAction(captureAction);
    expect(sharedEvent.type).toBe('ACTION_PROCESSED');

    const legacyAfter = legacy.getGameState();
    const sharedAfter = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedAfter)).toEqual(paritySnapshot(legacyAfter));
  });
});
describe('Move↔GameAction adapter – elimination parity scenario', () => {
  it('explicit elimination via eliminate_rings_from_stack + adapter keeps states aligned', async () => {
    const players = createPlayers();
    const boardType: BoardType = 'square8';

    const legacy = new LegacyGameEngine(
      'adapter-parity-eliminate',
      boardType,
      players,
      timeControl,
      true
    );
    legacy.startGame();

    const legacyAny: any = legacy;
    const boardManager = legacyAny.boardManager as any;
    const gameState = legacyAny.gameState as GameState;

    const stackPos: Position = { x: 0, y: 0 };
    const stack = {
      position: stackPos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };
    boardManager.setStack(stackPos, stack, gameState.board);

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'territory_processing';
    gameState.totalRingsEliminated = 0;
    gameState.board.eliminatedRings[1] = 0;
    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    p1.eliminatedRings = 0;

    const preLegacy = legacy.getGameState();
    const shared = new SharedGameEngine(preLegacy as unknown as SharedEngineGameState);

    const eliminationPayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'eliminate_rings_from_stack',
      player: 1,
      to: stackPos,
      thinkTime: 0,
    };

    const legacyResult = await legacy.makeMove(eliminationPayload);
    expect(legacyResult.success).toBe(true);

    const sharedBefore = shared.getGameState() as unknown as SharedEngineGameState;
    const fullMove: Move = {
      ...eliminationPayload,
      id: 'adapter-parity-elim',
      timestamp: new Date(),
      moveNumber: sharedBefore.moveHistory.length + 1,
    };

    const action = moveToGameAction(fullMove, sharedBefore);
    const event = shared.processAction(action);
    expect(event.type).toBe('ACTION_PROCESSED');

    const legacyAfter = legacy.getGameState();
    const sharedAfter = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedAfter)).toEqual(paritySnapshot(legacyAfter));
  });
});
