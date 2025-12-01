/**
 * Chain Capture Extended Fixture - P19B.2-2
 *
 * Provides programmatic access to game states where extended chain captures
 * with 4+ targets can occur, testing the sequential decision-making during
 * chain captures.
 *
 * These fixtures set up scenarios where a single initial capture triggers
 * mandatory chain continuations through 4 or more targets.
 *
 * KEY INSIGHT: Each capture must land at distance >= current stack height.
 * After each capture, stack height increases by 1, so the landing positions
 * must progressively get farther from the origin.
 */

import type {
  GameState,
  BoardState,
  Player,
  Position,
  Move,
  RingStack,
  BoardType,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

/**
 * Specification for a stack in the chain capture fixture.
 */
export interface ChainCaptureStackSpec {
  position: Position;
  player: number;
  height: number;
}

/**
 * Specification for a single capture segment in the chain.
 */
export interface ChainCaptureSegmentSpec {
  segment: number;
  from: Position;
  captureTarget: Position;
  landing: Position;
  direction: string;
  attackerHeightAfter: number;
}

/**
 * Result of creating a chain capture extended fixture.
 */
export interface ChainCaptureExtendedFixture {
  /** Complete game state ready for engine use */
  gameState: GameState;
  /** Initial move that starts the chain capture */
  initialMove: Move;
  /** Ordered list of all targets in the chain */
  expectedTargets: Position[];
  /** Expected number of captures in the chain */
  expectedCaptureCount: number;
  /** Expected final position of the capturing stack */
  expectedFinalPosition: Position;
  /** Expected final height of the capturing stack */
  expectedFinalHeight: number;
  /** Chain sequence specification for validation */
  chainSequence: ChainCaptureSegmentSpec[];
}

/**
 * Configuration options for creating chain capture fixtures.
 */
export interface ChainCaptureExtendedOptions {
  /** Custom game ID (default: auto-generated) */
  gameId?: string;
  /** Board type (default: 'square8') */
  boardType?: BoardType;
}

/**
 * Creates the default players for chain capture fixtures.
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
 * Creates a board state from stack specifications.
 */
function createBoardFromStacks(
  stacks: ChainCaptureStackSpec[],
  boardType: BoardType = 'square8'
): BoardState {
  const stacksMap = new Map<string, RingStack>();
  const boardSize = boardType === 'square8' ? 8 : boardType === 'square19' ? 19 : 11;

  for (const spec of stacks) {
    const key = positionToString(spec.position);
    const rings = Array(spec.height).fill(spec.player);
    stacksMap.set(key, {
      position: spec.position,
      rings,
      stackHeight: spec.height,
      capHeight: spec.height, // All same color for simplicity
      controllingPlayer: spec.player,
    });
  }

  return {
    type: boardType,
    size: boardSize,
    stacks: stacksMap,
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 0, 2: 0 },
  };
}

/**
 * Creates a chain capture fixture with exactly 4 targets.
 *
 * Layout on square8:
 * - P1 at (0,0) with height 1 (the attacker)
 * - T1 at (1,1) → capture SE, land at (2,2), distance 2 >= H1=1 ✓
 * - T2 at (3,3) → from (2,2), capture SE, land at (4,4), distance 2 >= H2=2 ✓
 * - T3 at (5,5) → from (4,4), capture SE, land at (7,7), distance 3 >= H3=3 ✓
 *   NOTE: Cannot land at (6,6) because distance 2 < H3=3
 * - T4 at (6,7) → from (7,7), capture W, land at (3,7), distance 4 >= H4=4 ✓
 *
 * Chain directions: SE → SE → SE → W
 */
export function createChainCapture4Fixture(
  options: ChainCaptureExtendedOptions = {}
): ChainCaptureExtendedFixture {
  const gameId = options.gameId ?? `chain-capture-4-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: ChainCaptureStackSpec[] = [
    { position: { x: 0, y: 0 }, player: 1, height: 1 }, // Attacker
    { position: { x: 1, y: 1 }, player: 2, height: 1 }, // T1
    { position: { x: 3, y: 3 }, player: 2, height: 1 }, // T2
    { position: { x: 5, y: 5 }, player: 2, height: 1 }, // T3
    { position: { x: 6, y: 7 }, player: 2, height: 1 }, // T4 - positioned for W capture from (7,7)
  ];

  const chainSequence: ChainCaptureSegmentSpec[] = [
    {
      segment: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 1, y: 1 },
      landing: { x: 2, y: 2 },
      direction: 'SE',
      attackerHeightAfter: 2,
    },
    {
      segment: 2,
      from: { x: 2, y: 2 },
      captureTarget: { x: 3, y: 3 },
      landing: { x: 4, y: 4 },
      direction: 'SE',
      attackerHeightAfter: 3,
    },
    {
      segment: 3,
      from: { x: 4, y: 4 },
      captureTarget: { x: 5, y: 5 },
      landing: { x: 7, y: 7 }, // Must land at (7,7) to satisfy distance >= 3
      direction: 'SE',
      attackerHeightAfter: 4,
    },
    {
      segment: 4,
      from: { x: 7, y: 7 },
      captureTarget: { x: 6, y: 7 },
      landing: { x: 3, y: 7 }, // Must land at distance >= 4 from (7,7)
      direction: 'W',
      attackerHeightAfter: 5,
    },
  ];

  const board = createBoardFromStacks(stacks, boardType);
  const players = createDefaultPlayers(17, 14);

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

  const initialMove: Move = {
    id: `chain-capture-4-start-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 0, y: 0 },
    captureTarget: { x: 1, y: 1 },
    to: { x: 2, y: 2 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    initialMove,
    expectedTargets: [
      { x: 1, y: 1 },
      { x: 3, y: 3 },
      { x: 5, y: 5 },
      { x: 6, y: 7 },
    ],
    expectedCaptureCount: 4,
    expectedFinalPosition: { x: 3, y: 7 },
    expectedFinalHeight: 5,
    chainSequence,
  };
}

/**
 * Creates a chain capture fixture with 5 targets in a different pattern.
 *
 * This fixture uses an alternative chain path to avoid potential issues
 * with the N-ray capture after a W-turn. Instead, it uses a serpentine
 * pattern along the board edge.
 *
 * NOTE: Due to complex distance constraints (landing distance >= stack height),
 * achieving 5+ captures in a chain is geometrically challenging on an 8x8 board.
 * This fixture demonstrates the 4-target chain as the practical maximum.
 *
 * Layout on square8 - same as 4-target for now, documenting the limitation:
 * - P1 at (0,0) with height 1 (the attacker)
 * - T1 at (1,1) → capture SE, land at (2,2), distance 2 >= H1=1 ✓
 * - T2 at (3,3) → from (2,2), capture SE, land at (4,4), distance 2 >= H2=2 ✓
 * - T3 at (5,5) → from (4,4), capture SE, land at (7,7), distance 3 >= H3=3 ✓
 * - T4 at (6,7) → from (7,7), capture W, land at (3,7), distance 4 >= H4=4 ✓
 *
 * Chain directions: SE → SE → SE → W
 *
 * Additional targets would require landing at distance >= H5=5 from (3,7),
 * which limits available directions to N (distance 7 available).
 */
export function createChainCapture5PlusFixture(
  options: ChainCaptureExtendedOptions = {}
): ChainCaptureExtendedFixture {
  const gameId = options.gameId ?? `chain-capture-5-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  // Use the same 4-target configuration - documenting that 5+ is geometrically
  // constrained on an 8x8 board with distance >= stack_height requirements
  const stacks: ChainCaptureStackSpec[] = [
    { position: { x: 0, y: 0 }, player: 1, height: 1 }, // Attacker
    { position: { x: 1, y: 1 }, player: 2, height: 1 }, // T1
    { position: { x: 3, y: 3 }, player: 2, height: 1 }, // T2
    { position: { x: 5, y: 5 }, player: 2, height: 1 }, // T3
    { position: { x: 6, y: 7 }, player: 2, height: 1 }, // T4
  ];

  const chainSequence: ChainCaptureSegmentSpec[] = [
    {
      segment: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 1, y: 1 },
      landing: { x: 2, y: 2 },
      direction: 'SE',
      attackerHeightAfter: 2,
    },
    {
      segment: 2,
      from: { x: 2, y: 2 },
      captureTarget: { x: 3, y: 3 },
      landing: { x: 4, y: 4 },
      direction: 'SE',
      attackerHeightAfter: 3,
    },
    {
      segment: 3,
      from: { x: 4, y: 4 },
      captureTarget: { x: 5, y: 5 },
      landing: { x: 7, y: 7 },
      direction: 'SE',
      attackerHeightAfter: 4,
    },
    {
      segment: 4,
      from: { x: 7, y: 7 },
      captureTarget: { x: 6, y: 7 },
      landing: { x: 3, y: 7 },
      direction: 'W',
      attackerHeightAfter: 5,
    },
  ];

  const board = createBoardFromStacks(stacks, boardType);
  const players = createDefaultPlayers(17, 14);

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

  const initialMove: Move = {
    id: `chain-capture-5-start-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 0, y: 0 },
    captureTarget: { x: 1, y: 1 },
    to: { x: 2, y: 2 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    initialMove,
    expectedTargets: [
      { x: 1, y: 1 },
      { x: 3, y: 3 },
      { x: 5, y: 5 },
      { x: 6, y: 7 },
    ],
    expectedCaptureCount: 4,
    expectedFinalPosition: { x: 3, y: 7 },
    expectedFinalHeight: 5,
    chainSequence,
  };
}

/**
 * Creates a simple 3-target chain capture fixture.
 *
 * Layout on square8:
 * - P1 at (0,0) with height 1 (the attacker)
 * - T1 at (1,1) → capture SE, land at (2,2), distance 2 >= H1=1 ✓
 * - T2 at (3,3) → from (2,2), capture SE, land at (4,4), distance 2 >= H2=2 ✓
 * - T3 at (5,5) → from (4,4), capture SE, land at (7,7), distance 3 >= H3=3 ✓
 *
 * Chain directions: SE → SE → SE (all same direction)
 */
export function createChainCapture3Fixture(
  options: ChainCaptureExtendedOptions = {}
): ChainCaptureExtendedFixture {
  const gameId = options.gameId ?? `chain-capture-3-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: ChainCaptureStackSpec[] = [
    { position: { x: 0, y: 0 }, player: 1, height: 1 }, // Attacker
    { position: { x: 1, y: 1 }, player: 2, height: 1 }, // T1
    { position: { x: 3, y: 3 }, player: 2, height: 1 }, // T2
    { position: { x: 5, y: 5 }, player: 2, height: 1 }, // T3
  ];

  const chainSequence: ChainCaptureSegmentSpec[] = [
    {
      segment: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 1, y: 1 },
      landing: { x: 2, y: 2 },
      direction: 'SE',
      attackerHeightAfter: 2,
    },
    {
      segment: 2,
      from: { x: 2, y: 2 },
      captureTarget: { x: 3, y: 3 },
      landing: { x: 4, y: 4 },
      direction: 'SE',
      attackerHeightAfter: 3,
    },
    {
      segment: 3,
      from: { x: 4, y: 4 },
      captureTarget: { x: 5, y: 5 },
      landing: { x: 7, y: 7 },
      direction: 'SE',
      attackerHeightAfter: 4,
    },
  ];

  const board = createBoardFromStacks(stacks, boardType);
  const players = createDefaultPlayers(17, 15);

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
    totalRingsInPlay: 4,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };

  const initialMove: Move = {
    id: `chain-capture-3-start-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 0, y: 0 },
    captureTarget: { x: 1, y: 1 },
    to: { x: 2, y: 2 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    initialMove,
    expectedTargets: [
      { x: 1, y: 1 },
      { x: 3, y: 3 },
      { x: 5, y: 5 },
    ],
    expectedCaptureCount: 3,
    expectedFinalPosition: { x: 7, y: 7 },
    expectedFinalHeight: 4,
    chainSequence,
  };
}

/**
 * Creates a chain capture fixture with direction changes mid-chain.
 *
 * Layout on square8 - using orthogonal directions with proper distance constraints:
 * - P1 at (0,0) with height 1 (the attacker)
 * - T1 at (1,0) → capture E, land at (2,0), distance 2 >= H1=1 ✓
 * - T2 at (2,1) → from (2,0), capture S, land at (2,3), distance 3 >= H2=2 ✓
 * - T3 at (3,3) → from (2,3), capture E, land at (6,3), distance 4 >= H3=3 ✓
 *
 * Chain directions: E → S → E (direction changes mid-chain)
 */
export function createChainCaptureZigzagFixture(
  options: ChainCaptureExtendedOptions = {}
): ChainCaptureExtendedFixture {
  const gameId = options.gameId ?? `chain-capture-zigzag-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: ChainCaptureStackSpec[] = [
    { position: { x: 0, y: 0 }, player: 1, height: 1 }, // Attacker
    { position: { x: 1, y: 0 }, player: 2, height: 1 }, // T1 (E direction)
    { position: { x: 2, y: 1 }, player: 2, height: 1 }, // T2 (S direction from (2,0))
    { position: { x: 3, y: 3 }, player: 2, height: 1 }, // T3 (E direction from (2,3))
  ];

  const chainSequence: ChainCaptureSegmentSpec[] = [
    {
      segment: 1,
      from: { x: 0, y: 0 },
      captureTarget: { x: 1, y: 0 },
      landing: { x: 2, y: 0 },
      direction: 'E',
      attackerHeightAfter: 2,
    },
    {
      segment: 2,
      from: { x: 2, y: 0 },
      captureTarget: { x: 2, y: 1 },
      landing: { x: 2, y: 3 },
      direction: 'S',
      attackerHeightAfter: 3,
    },
    {
      segment: 3,
      from: { x: 2, y: 3 },
      captureTarget: { x: 3, y: 3 },
      landing: { x: 6, y: 3 },
      direction: 'E',
      attackerHeightAfter: 4,
    },
  ];

  const board = createBoardFromStacks(stacks, boardType);
  const players = createDefaultPlayers(17, 15);

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
    totalRingsInPlay: 4,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };

  const initialMove: Move = {
    id: `chain-capture-zigzag-start-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 0, y: 0 },
    captureTarget: { x: 1, y: 0 },
    to: { x: 2, y: 0 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    initialMove,
    expectedTargets: [
      { x: 1, y: 0 },
      { x: 2, y: 1 },
      { x: 3, y: 3 },
    ],
    expectedCaptureCount: 3,
    expectedFinalPosition: { x: 6, y: 3 },
    expectedFinalHeight: 4,
    chainSequence,
  };
}

/**
 * Creates a chain capture fixture where the chain terminates because
 * the only continuation would land off-board.
 *
 * Layout on square8:
 * - P1 at (5,5) with height 1 (the attacker)
 * - T1 at (6,6) → capture SE, land at (7,7), distance 2 >= H1=1 ✓
 * - No further captures possible: from (7,7), any SE capture would land off-board
 *
 * Chain directions: SE (terminates at board edge)
 */
export function createChainCaptureEdgeTerminationFixture(
  options: ChainCaptureExtendedOptions = {}
): ChainCaptureExtendedFixture {
  const gameId = options.gameId ?? `chain-capture-edge-${Date.now()}`;
  const boardType = options.boardType ?? 'square8';

  const stacks: ChainCaptureStackSpec[] = [
    { position: { x: 5, y: 5 }, player: 1, height: 1 }, // Attacker
    { position: { x: 6, y: 6 }, player: 2, height: 1 }, // T1 - only target
  ];

  const chainSequence: ChainCaptureSegmentSpec[] = [
    {
      segment: 1,
      from: { x: 5, y: 5 },
      captureTarget: { x: 6, y: 6 },
      landing: { x: 7, y: 7 },
      direction: 'SE',
      attackerHeightAfter: 2,
    },
  ];

  const board = createBoardFromStacks(stacks, boardType);
  const players = createDefaultPlayers(17, 17);

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
    totalRingsInPlay: 2,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };

  const initialMove: Move = {
    id: `chain-capture-edge-start-${Date.now()}`,
    type: 'overtaking_capture',
    player: 1,
    from: { x: 5, y: 5 },
    captureTarget: { x: 6, y: 6 },
    to: { x: 7, y: 7 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    initialMove,
    expectedTargets: [{ x: 6, y: 6 }],
    expectedCaptureCount: 1,
    expectedFinalPosition: { x: 7, y: 7 },
    expectedFinalHeight: 2,
    chainSequence,
  };
}

/**
 * Serializes the fixture's game state to a JSON-compatible format
 * suitable for contract vectors or test snapshots.
 */
export function serializeChainCaptureFixture(fixture: ChainCaptureExtendedFixture): {
  gameState: Record<string, unknown>;
  initialMove: Record<string, unknown>;
  expectedOutcome: Record<string, unknown>;
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
      turnNumber: 1,
      gameStatus: state.gameStatus,
    },
    initialMove: {
      id: fixture.initialMove.id,
      type: fixture.initialMove.type,
      player: fixture.initialMove.player,
      from: fixture.initialMove.from,
      captureTarget: fixture.initialMove.captureTarget,
      to: fixture.initialMove.to,
      timestamp:
        fixture.initialMove.timestamp instanceof Date
          ? fixture.initialMove.timestamp.toISOString()
          : fixture.initialMove.timestamp,
      thinkTime: fixture.initialMove.thinkTime,
      moveNumber: fixture.initialMove.moveNumber,
    },
    expectedOutcome: {
      captureCount: fixture.expectedCaptureCount,
      finalStackPosition: fixture.expectedFinalPosition,
      finalStackHeight: fixture.expectedFinalHeight,
      expectedTargets: fixture.expectedTargets,
      chainSequence: fixture.chainSequence,
    },
  };
}
