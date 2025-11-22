import { GameEngine as SharedGameEngine } from '../../src/shared/engine/GameEngine';
import {
  GameState as SharedEngineGameState,
  PlaceRingAction,
  MoveStackAction,
} from '../../src/shared/engine/types';
import { GameEngine as LegacyGameEngine } from '../../src/server/game/GameEngine';
import { BoardType, GameState, Move, Player } from '../../src/shared/types/game';
import {
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../src/shared/engine/core';

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

function stripMoveForComparison(move: Move) {
  // Ignore non-semantic fields when comparing membership.
  const { id, timestamp, thinkTime, moveNumber, ...rest } = move;
  return rest;
}

describe('Refactored shared GameEngine vs legacy backend GameEngine parity (minimal)', () => {
  it('placement + movement sequence keeps states and invariants in lockstep', async () => {
    const { legacy, shared } = createEngines('parity-basic');

    const legacyState0 = legacy.getGameState();
    const sharedState0 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState0)).toEqual(paritySnapshot(legacyState0));

    // Step 1: player 1 places a single ring at (0,0).
    const placeMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      thinkTime: 0,
    };

    const placementResult = await legacy.makeMove(placeMove);
    expect(placementResult.success).toBe(true);

    const placeAction: PlaceRingAction = {
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 1,
    };
    const sharedEvent1 = shared.processAction(placeAction);
    expect(sharedEvent1.type).toBe('ACTION_PROCESSED');

    const legacyState1 = legacy.getGameState();
    const sharedState1 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState1)).toEqual(paritySnapshot(legacyState1));

    // Step 2: same player moves stack from (0,0) to (0,1).
    const movePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      type: 'move_stack',
      player: legacyState1.currentPlayer,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
      thinkTime: 0,
    };

    const moveResult = await legacy.makeMove(movePayload);
    expect(moveResult.success).toBe(true);

    const moveAction: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
    };
    const sharedEvent2 = shared.processAction(moveAction);
    expect(sharedEvent2.type).toBe('ACTION_PROCESSED');

    const legacyState2 = legacy.getGameState();
    const sharedState2 = shared.getGameState() as unknown as GameState;
    expect(paritySnapshot(sharedState2)).toEqual(paritySnapshot(legacyState2));
  });

  it('canonical initial placement move is considered legal by both engines', () => {
    const { legacy, shared } = createEngines('parity-legality');
    const legacyState = legacy.getGameState();
    const sharedState = shared.getGameState() as unknown as GameState;

    const candidateMove: Move = {
      id: 'p1-0,0',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: legacyState.moveHistory.length + 1,
    };

    const legacyMoves = legacy.getValidMoves(1);
    const hasCandidate = legacyMoves.some(
      (m) =>
        m.type === 'place_ring' &&
        m.player === candidateMove.player &&
        m.to &&
        m.to.x === candidateMove.to.x &&
        m.to.y === candidateMove.to.y
    );
    expect(hasCandidate).toBe(true);

    const sharedAction: PlaceRingAction = {
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 1,
    };

    // Determine legality by running shared engine on a throwaway clone.
    const tempShared = new SharedGameEngine(sharedState as unknown as SharedEngineGameState);
    const event = tempShared.processAction(sharedAction);
    expect(event.type).toBe('ACTION_PROCESSED');
  });
});
