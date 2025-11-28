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

// Skip this test suite when orchestrator adapter is enabled - phase divergence between territory_processing vs ring_placement (intentional)
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

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
  // Enable move-driven phases to match SharedEngine behavior
  legacy.enableMoveDrivenDecisionPhases();
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

(skipWithOrchestrator ? describe.skip : describe)(
  'Refactored shared GameEngine vs legacy backend GameEngine parity (minimal)',
  () => {
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

    it('territory disconnection triggers correct phase transition in both engines', async () => {
      const { legacy, shared } = createEngines('parity-territory');
      const legacyState = legacy.getGameState();

      // Setup: Create a potential disconnected region
      // We need to manually manipulate the board to set up the scenario quickly
      // P1 surrounds (0,0) with markers/collapsed spaces
      const board = legacyState.board;
      const p1 = 1;

      // Place markers to surround (0,0)
      // (0,1), (1,0), (1,1)
      // We use the boardManager from legacy engine to set this up correctly
      const legacyAny: any = legacy;
      const boardManager = legacyAny.boardManager;

      // Access internal state directly to modify it
      const internalBoard = legacyAny.gameState.board;

      boardManager.setMarker({ x: 0, y: 1 }, p1, internalBoard);
      boardManager.setMarker({ x: 1, y: 0 }, p1, internalBoard);
      boardManager.setMarker({ x: 1, y: 1 }, p1, internalBoard);

      // Ensure P1 has a stack outside to satisfy self-elimination prerequisite
      // Place stack at (3,3)
      const stack = {
        position: { x: 3, y: 3 },
        rings: [p1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: p1,
      };
      boardManager.setStack({ x: 3, y: 3 }, stack, internalBoard);

      // Sync shared engine with this state
      const syncedState = legacy.getGameState();
      const sharedAny: any = shared;
      sharedAny.state = JSON.parse(JSON.stringify(syncedState)); // Deep copy to ensure clean state

      // Now P1 moves a stack to (0,0) to complete the disconnection?
      // No, usually you place a marker to complete the border.
      // Let's say P1 moves from (0,2) to (0,1) (which is already marked? No).
      // Let's say (0,0) is the region.
      // Border is (0,1), (1,1), (1,0).
      // If P1 moves to (0,1) and leaves a marker at (0,2), that doesn't help.

      // Better setup:
      // Region is (0,0).
      // Border needs to be closed.
      // Existing markers: (1,0), (1,1).
      // P1 moves from (0,2) to (0,1).
      // Leaves marker at (0,2).
      // Lands at (0,1).
      // This creates a marker at (0,2).
      // Wait, we need to surround (0,0).
      // Neighbors of (0,0) are (0,1) and (1,0).
      // If (1,0) has a marker.
      // And P1 moves FROM (0,0) to somewhere else? No, (0,0) is the region.

      // Let's try:
      // Region: (0,0).
      // P1 has markers at (1,0) and (1,1).
      // P1 moves a stack from (0,2) to (0,1).
      // This leaves a marker at (0,2).
      // And lands at (0,1).
      // Now (0,0) is surrounded by:
      // (1,0) - marker
      // (0,1) - stack (acts as barrier? No, stack is not a marker border).

      // Actually, a region is disconnected if surrounded by markers/collapsed/edges.
      // So we need markers at (0,1) and (1,0).
      // Setup: Marker at (1,0).
      // P1 moves from (0,1) to (0,2).
      // Leaves marker at (0,1).
      // Now (0,0) is surrounded by (0,1) and (1,0) and edges.
      // (0,0) is empty.
      // It lacks representation (no stacks).
      // So it should disconnect.

      // Setup for move:
      // Stack at (0,1).
      // Marker at (1,0).
      // Move (0,1) -> (0,2).

      boardManager.setMarker({ x: 1, y: 0 }, p1, internalBoard);
      // Clear other markers from previous attempt
      internalBoard.markers.delete('0,1');
      internalBoard.markers.delete('1,1');

      const moveStack = {
        position: { x: 0, y: 1 },
        rings: [p1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: p1,
      };
      boardManager.setStack({ x: 0, y: 1 }, moveStack, internalBoard);

      // Force phase to movement
      legacyAny.gameState.currentPhase = 'movement';

      // Sync again
      const syncedState2 = legacy.getGameState();
      // Manually reconstruct Maps because JSON.parse/stringify destroys them
      const sharedState2 = {
        ...syncedState2,
        board: {
          ...syncedState2.board,
          stacks: new Map(syncedState2.board.stacks),
          markers: new Map(syncedState2.board.markers),
          collapsedSpaces: new Map(syncedState2.board.collapsedSpaces),
          territories: new Map(syncedState2.board.territories),
        },
      };
      sharedAny.state = sharedState2;

      // Execute move on Legacy
      const movePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
        type: 'move_stack',
        player: p1,
        from: { x: 0, y: 1 },
        to: { x: 0, y: 2 },
        thinkTime: 0,
      };

      const legacyResult = await legacy.makeMove(movePayload);
      if (!legacyResult.success) {
        console.error('Legacy Move Failed:', legacyResult.error);
      }
      expect(legacyResult.success).toBe(true);

      // Execute move on Shared
      const moveAction: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: p1,
        from: { x: 0, y: 1 },
        to: { x: 0, y: 2 },
      };
      const sharedEvent = shared.processAction(moveAction);
      expect(sharedEvent.type).toBe('ACTION_PROCESSED');

      // Verify Phase Transition
      // Legacy engine should transition to 'territory_processing' (or handle it if automatic)
      // Shared engine should transition to 'territory_processing'

      const legacyStateAfter = legacy.getGameState();
      const sharedStateAfter = shared.getGameState() as unknown as GameState;

      // Note: Legacy engine might auto-process if configured, or wait in territory_processing.
      // The default GameEngine processes automatic consequences immediately unless move-driven phases are enabled.
      // So legacyStateAfter.currentPhase might be 'movement' (next player) or 'territory_processing'.
      // However, SharedGameEngine.checkStateTransitions sets phase to 'territory_processing' if regions found.

      // Let's check if they match.
      // If Legacy auto-processed, it will be in next turn.
      // If Shared DOES NOT auto-process (it just changes phase), then they will mismatch.

      // This highlights a potential divergence: Legacy handles auto-processing internally, Shared might expect explicit actions.
      // If so, we need to align them.

      expect(sharedStateAfter.currentPhase).toEqual(legacyStateAfter.currentPhase);
    });
  }
);
