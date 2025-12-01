/**
 * Multi-Phase Turn Fixture - P19B.2-3
 *
 * Provides programmatic access to game states where a single action
 * triggers multiple turn phases in sequence:
 * - placement → movement/capture → chain_capture → line_processing → territory_processing
 *
 * These fixtures test the complete turn lifecycle where each phase
 * flows correctly into the next based on game state changes.
 */

import type {
  GameState,
  BoardState,
  Player,
  Position,
  Move,
  RingStack,
  BoardType,
  GamePhase,
  MarkerInfo,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

/**
 * Specification for a stack in the multi-phase turn fixture.
 */
export interface MultiPhaseStackSpec {
  position: Position;
  player: number;
  height: number;
}

/**
 * Specification for a marker in the fixture.
 */
export interface MultiPhaseMarkerSpec {
  position: Position;
  player: number;
}

/**
 * Phase transition information for verification.
 */
export interface PhaseTransition {
  phaseAfter: GamePhase;
  description: string;
  stackPosition?: Position;
  stackHeight?: number;
  availableChainTarget?: Position;
  expectedLandingPosition?: Position;
  linePositions?: Position[];
}

/**
 * Expected state after each phase completes.
 */
export interface PhaseExpectation {
  phase: GamePhase;
  stackCount?: number;
  markerCount?: number;
  collapsedCount?: number;
  player1TerritorySpaces?: number;
  player2TerritorySpaces?: number;
}

/**
 * Result of creating a multi-phase turn fixture.
 */
export interface MultiPhaseTurnFixture {
  /** Complete game state ready for engine use */
  gameState: GameState;
  /** Initial triggering action (move/capture) */
  triggeringAction: Move;
  /** Ordered list of phases expected during this turn */
  expectedPhaseSequence: GamePhase[];
  /** Detailed expectations after each phase */
  phaseExpectations: {
    afterPlacement?: Partial<GameState>;
    afterCapture?: Partial<GameState>;
    afterChainCapture?: Partial<GameState>;
    afterLine?: Partial<GameState>;
    afterTerritory?: Partial<GameState>;
  };
  /** Phase transition details for validation */
  phaseTransitions: PhaseTransition[];
  /** Expected final state after all phases complete */
  expectedFinalState: {
    currentPlayer: number;
    currentPhase: GamePhase;
    gameStatus: 'active' | 'completed';
  };
}

/**
 * Configuration options for creating multi-phase turn fixtures.
 */
export interface MultiPhaseTurnOptions {
  /** Custom game ID (default: auto-generated) */
  gameId?: string;
  /** Board type (default: 'square8') */
  boardType?: BoardType;
}

/**
 * Creates the default players for multi-phase fixtures.
 */
function createDefaultPlayers(ringsInHandP1: number, ringsInHandP2: number): Player[] {
  return [
    {
      id: 'player-1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: ringsInHandP1,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'player-2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: ringsInHandP2,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

/**
 * Creates a board state from stack and marker specifications.
 */
function createBoardFromSpecs(
  stacks: MultiPhaseStackSpec[],
  markers: MultiPhaseMarkerSpec[],
  boardType: BoardType = 'square8'
): BoardState {
  const stacksMap = new Map<string, RingStack>();
  const markersMap = new Map<string, MarkerInfo>();
  const boardSize = boardType === 'square8' ? 8 : boardType === 'square19' ? 19 : 11;

  for (const spec of stacks) {
    const key = positionToString(spec.position);
    const rings = Array(spec.height).fill(spec.player);
    stacksMap.set(key, {
      position: spec.position,
      rings,
      stackHeight: spec.height,
      capHeight: spec.height,
      controllingPlayer: spec.player,
    });
  }

  for (const spec of markers) {
    const key = positionToString(spec.position);
    markersMap.set(key, {
      position: spec.position,
      player: spec.player,
      type: 'regular',
    });
  }

  return {
    type: boardType,
    size: boardSize,
    stacks: stacksMap,
    markers: markersMap,
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 0, 2: 0 },
  };
}

/**
 * Creates a multi-phase turn fixture that triggers:
 * movement → chain_capture → line_processing → territory_processing
 *
 * Scenario:
 * - Player 1 has a stack at (0,3) that can capture at (1,3)
 * - After capture, chain capture available at (3,3)
 * - Final landing position completes a line with existing markers
 * - Turn then proceeds through territory processing
 *
 * Layout on square8:
 * - P1 stack at (0,3) height 1
 * - P2 stack at (1,3) height 1 (capture target 1)
 * - P2 stack at (3,3) height 1 (chain capture target)
 * - P1 stack at (7,3) height 2 (anchoring stack)
 * - P1 markers at (4,3), (5,3), (6,3) forming partial line
 *
 * After capture chain lands at (4,3) or creates markers completing the line
 */
export function createMultiPhaseTurnFixture(
  options: MultiPhaseTurnOptions = {}
): MultiPhaseTurnFixture {
  const gameId = options.gameId ?? `multi-phase-turn-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: MultiPhaseStackSpec[] = [
    { position: { x: 0, y: 3 }, player: 1, height: 1 },
    { position: { x: 1, y: 3 }, player: 2, height: 1 },
    { position: { x: 3, y: 3 }, player: 2, height: 1 },
    { position: { x: 7, y: 3 }, player: 1, height: 2 },
  ];

  const markers: MultiPhaseMarkerSpec[] = [
    { position: { x: 4, y: 3 }, player: 1 },
    { position: { x: 5, y: 3 }, player: 1 },
    { position: { x: 6, y: 3 }, player: 1 },
  ];

  const phaseTransitions: PhaseTransition[] = [
    {
      phaseAfter: 'chain_capture',
      description: 'After first capture at (1,3), chain capture detected at (3,3)',
      stackPosition: { x: 2, y: 3 },
      stackHeight: 2,
      availableChainTarget: { x: 3, y: 3 },
    },
    {
      phaseAfter: 'line_processing',
      description: 'After chain capture completes, line detection runs',
      expectedLandingPosition: { x: 4, y: 3 },
      linePositions: [
        { x: 0, y: 3 },
        { x: 1, y: 3 },
        { x: 2, y: 3 },
        { x: 3, y: 3 },
        { x: 4, y: 3 },
      ],
    },
    {
      phaseAfter: 'territory_processing',
      description: 'After line processing, territory check for disconnected regions',
    },
  ];

  const board = createBoardFromSpecs(stacks, markers, boardType);
  const players = createDefaultPlayers(14, 16);

  const gameState: GameState = {
    id: gameId,
    boardType,
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: {
      initialTime: 600,
      increment: 0,
      type: 'blitz',
    },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 5,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };

  const triggeringAction: Move = {
    id: `multi-phase-trigger-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 0, y: 3 },
    captureTarget: { x: 1, y: 3 },
    to: { x: 2, y: 3 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 5,
  };

  return {
    gameState,
    triggeringAction,
    expectedPhaseSequence: ['movement', 'chain_capture', 'line_processing', 'territory_processing'],
    phaseExpectations: {
      afterCapture: {
        currentPhase: 'chain_capture',
        currentPlayer: 1,
      },
      afterChainCapture: {
        currentPhase: 'line_processing',
        currentPlayer: 1,
      },
      afterLine: {
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      },
      afterTerritory: {
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      },
    },
    phaseTransitions,
    expectedFinalState: {
      currentPlayer: 2,
      currentPhase: 'ring_placement',
      gameStatus: 'active',
    },
  };
}

/**
 * Creates a full sequence multi-phase turn fixture that attempts
 * to trigger all major phases including territory disconnection.
 *
 * Scenario:
 * - Player 1 captures triggering a chain
 * - Chain completes a line
 * - Line completion disconnects a corner region
 * - Territory processing handles the disconnection
 *
 * This is a more complex scenario designed to test the full
 * turn lifecycle through all phases.
 */
export function createFullSequenceTurnFixture(
  options: MultiPhaseTurnOptions = {}
): MultiPhaseTurnFixture {
  const gameId = options.gameId ?? `full-sequence-turn-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: MultiPhaseStackSpec[] = [
    { position: { x: 1, y: 1 }, player: 1, height: 1 },
    { position: { x: 2, y: 1 }, player: 2, height: 1 },
    { position: { x: 4, y: 1 }, player: 2, height: 1 },
    { position: { x: 0, y: 0 }, player: 2, height: 1 }, // Corner stack that may be disconnected
    { position: { x: 7, y: 7 }, player: 1, height: 1 }, // Anchor for P1
  ];

  const markers: MultiPhaseMarkerSpec[] = [
    { position: { x: 0, y: 1 }, player: 1 },
    { position: { x: 5, y: 1 }, player: 1 },
    { position: { x: 6, y: 1 }, player: 1 },
    { position: { x: 7, y: 1 }, player: 1 },
    { position: { x: 1, y: 0 }, player: 2 }, // P2 marker near corner
  ];

  const phaseTransitions: PhaseTransition[] = [
    {
      phaseAfter: 'chain_capture',
      description: 'After capture at (2,1), chain opportunity at (4,1)',
      stackPosition: { x: 3, y: 1 },
      stackHeight: 2,
      availableChainTarget: { x: 4, y: 1 },
    },
    {
      phaseAfter: 'line_processing',
      description: 'Chain lands at (5,1), line of 5 markers formed in row 1',
      expectedLandingPosition: { x: 5, y: 1 },
    },
    {
      phaseAfter: 'territory_processing',
      description: 'Line may disconnect corner region at (0,0)',
    },
  ];

  const board = createBoardFromSpecs(stacks, markers, boardType);
  const players = createDefaultPlayers(14, 15);

  const gameState: GameState = {
    id: gameId,
    boardType,
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: {
      initialTime: 600,
      increment: 0,
      type: 'blitz',
    },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 5,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };

  const triggeringAction: Move = {
    id: `full-sequence-trigger-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 1, y: 1 },
    captureTarget: { x: 2, y: 1 },
    to: { x: 3, y: 1 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 6,
  };

  return {
    gameState,
    triggeringAction,
    expectedPhaseSequence: ['movement', 'chain_capture', 'line_processing', 'territory_processing'],
    phaseExpectations: {
      afterCapture: {
        currentPhase: 'chain_capture',
        currentPlayer: 1,
      },
      afterChainCapture: {
        currentPhase: 'line_processing',
        currentPlayer: 1,
      },
      afterLine: {
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      },
      afterTerritory: {
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      },
    },
    phaseTransitions,
    expectedFinalState: {
      currentPlayer: 2,
      currentPhase: 'ring_placement',
      gameStatus: 'active',
    },
  };
}

/**
 * Creates a simple placement-to-capture multi-phase fixture.
 *
 * Scenario:
 * - Player 1 places a ring on an existing stack (ring_placement)
 * - This creates a must-move situation (movement)
 * - Movement has capture opportunities
 *
 * Tests the ring_placement → movement phase transition.
 */
export function createPlacementToMovementFixture(
  options: MultiPhaseTurnOptions = {}
): MultiPhaseTurnFixture {
  const gameId = options.gameId ?? `placement-to-movement-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: MultiPhaseStackSpec[] = [
    { position: { x: 3, y: 3 }, player: 1, height: 1 }, // Stack to place on
    { position: { x: 3, y: 4 }, player: 2, height: 1 }, // Potential capture target
    { position: { x: 5, y: 5 }, player: 2, height: 1 }, // Another P2 stack
  ];

  const markers: MultiPhaseMarkerSpec[] = [];

  const phaseTransitions: PhaseTransition[] = [
    {
      phaseAfter: 'movement',
      description:
        'After placement on (3,3), stack becomes height 2 and must move with capture available',
      stackPosition: { x: 3, y: 3 },
      stackHeight: 2,
    },
    {
      phaseAfter: 'line_processing',
      description: 'After movement/capture, line detection runs',
    },
    {
      phaseAfter: 'territory_processing',
      description: 'After line processing, territory check',
    },
  ];

  const board = createBoardFromSpecs(stacks, markers, boardType);
  const players = createDefaultPlayers(15, 16);

  const gameState: GameState = {
    id: gameId,
    boardType,
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: {
      initialTime: 600,
      increment: 0,
      type: 'blitz',
    },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 3,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };

  const triggeringAction: Move = {
    id: `placement-trigger-${Date.now()}`,
    type: 'place_ring',
    player: 1,
    to: { x: 3, y: 3 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 4,
  };

  return {
    gameState,
    triggeringAction,
    expectedPhaseSequence: [
      'ring_placement',
      'movement',
      'line_processing',
      'territory_processing',
    ],
    phaseExpectations: {
      afterPlacement: {
        currentPhase: 'movement',
        currentPlayer: 1,
      },
      afterCapture: {
        currentPhase: 'line_processing',
        currentPlayer: 1,
      },
      afterLine: {
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      },
      afterTerritory: {
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      },
    },
    phaseTransitions,
    expectedFinalState: {
      currentPlayer: 2,
      currentPhase: 'ring_placement',
      gameStatus: 'active',
    },
  };
}

/**
 * Serializes the fixture's game state to a JSON-compatible format
 * suitable for contract vectors or test snapshots.
 */
export function serializeMultiPhaseTurnFixture(fixture: MultiPhaseTurnFixture): {
  gameState: Record<string, unknown>;
  triggeringAction: Record<string, unknown>;
  expectedPhaseSequence: string[];
  expectedFinalState: Record<string, unknown>;
} {
  const state = fixture.gameState;

  // Convert Maps to plain objects for JSON serialization
  const stacksObj: Record<string, unknown> = {};
  for (const [key, stack] of state.board.stacks) {
    stacksObj[key] = {
      position: stack.position,
      rings: stack.rings,
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  }

  const markersObj: Record<string, unknown> = {};
  for (const [key, marker] of state.board.markers) {
    markersObj[key] = {
      player: marker.player,
      position: marker.position,
      type: marker.type,
    };
  }

  const collapsedObj: Record<string, number> = {};
  for (const [key, owner] of state.board.collapsedSpaces) {
    collapsedObj[key] = owner;
  }

  return {
    gameState: {
      gameId: state.id,
      boardType: state.boardType,
      board: {
        type: state.board.type,
        size: state.board.size,
        stacks: stacksObj,
        markers: markersObj,
        collapsedSpaces: collapsedObj,
        eliminatedRings: state.board.eliminatedRings,
      },
      players: state.players.map((p) => ({
        playerNumber: p.playerNumber,
        ringsInHand: p.ringsInHand,
        eliminatedRings: p.eliminatedRings,
        territorySpaces: p.territorySpaces,
        isActive: true,
      })),
      currentPlayer: state.currentPlayer,
      currentPhase: state.currentPhase,
      turnNumber: 5,
      gameStatus: state.gameStatus,
      victoryThreshold: state.victoryThreshold,
      territoryVictoryThreshold: state.territoryVictoryThreshold,
    },
    triggeringAction: {
      id: fixture.triggeringAction.id,
      type: fixture.triggeringAction.type,
      player: fixture.triggeringAction.player,
      from: fixture.triggeringAction.from,
      captureTarget: fixture.triggeringAction.captureTarget,
      to: fixture.triggeringAction.to,
      timestamp:
        fixture.triggeringAction.timestamp instanceof Date
          ? fixture.triggeringAction.timestamp.toISOString()
          : fixture.triggeringAction.timestamp,
      thinkTime: fixture.triggeringAction.thinkTime,
      moveNumber: fixture.triggeringAction.moveNumber,
    },
    expectedPhaseSequence: fixture.expectedPhaseSequence,
    expectedFinalState: fixture.expectedFinalState,
  };
}
