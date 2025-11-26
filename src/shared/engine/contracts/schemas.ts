/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Contract Schemas for Cross-Engine Parity
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This file defines the JSON Schema contracts used for parity testing between
 * the TypeScript canonical engine and the Python AI rules engine.
 *
 * These schemas serve as the single source of truth for:
 * 1. Request/response formats for engine API calls
 * 2. Test vector validation
 * 3. Runtime contract enforcement
 */

// ═══════════════════════════════════════════════════════════════════════════
// Position Schema
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Position in board coordinates.
 * For square boards: {x, y}
 * For hexagonal boards: {x, y, z} using cube coordinates
 */
export const PositionSchema = {
  $id: 'Position',
  type: 'object',
  properties: {
    x: { type: 'integer', description: 'X coordinate (column for square, q for hex)' },
    y: { type: 'integer', description: 'Y coordinate (row for square, r for hex)' },
    z: { type: 'integer', description: 'Z coordinate (s for hex cube coords, -x-y)' },
  },
  required: ['x', 'y'],
  additionalProperties: false,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Move Schemas
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Move type enumeration.
 */
export const MoveTypeSchema = {
  $id: 'MoveType',
  type: 'string',
  enum: [
    'place_ring',
    'skip_placement',
    'move_stack',
    'move_ring',
    'overtaking_capture',
    'continue_capture_segment',
    'process_line',
    'choose_line_reward',
    'process_territory_region',
    'eliminate_rings_from_stack',
    'pass',
    'resign',
  ],
} as const;

/**
 * Move object for engine API.
 */
export const MoveSchema = {
  $id: 'Move',
  type: 'object',
  properties: {
    id: { type: 'string', description: 'Unique move identifier' },
    type: { $ref: 'MoveType' },
    player: { type: 'integer', minimum: 1, description: 'Player number' },
    from: { $ref: 'Position', description: 'Source position (for movement/capture)' },
    to: { $ref: 'Position', description: 'Target position' },
    captureTarget: { $ref: 'Position', description: 'Captured stack position' },
    placementCount: { type: 'integer', minimum: 1, maximum: 3 },
    moveNumber: { type: 'integer', minimum: 1 },
    timestamp: { type: 'string', format: 'date-time' },
    thinkTime: { type: 'number', minimum: 0 },
  },
  required: ['type', 'player', 'to'],
  additionalProperties: true,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Board State Schemas
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Ring stack representation.
 */
export const RingStackSchema = {
  $id: 'RingStack',
  type: 'object',
  properties: {
    position: { $ref: 'Position' },
    rings: {
      type: 'array',
      items: { type: 'integer', minimum: 1 },
      description: 'Ring owners from top to bottom (index 0 = top)',
    },
    stackHeight: { type: 'integer', minimum: 0 },
    capHeight: { type: 'integer', minimum: 0 },
    controllingPlayer: { type: 'integer', minimum: 1 },
  },
  required: ['position', 'rings', 'stackHeight', 'capHeight', 'controllingPlayer'],
  additionalProperties: false,
} as const;

/**
 * Marker representation.
 */
export const MarkerSchema = {
  $id: 'Marker',
  type: 'object',
  properties: {
    position: { $ref: 'Position' },
    player: { type: 'integer', minimum: 1 },
    type: { type: 'string', enum: ['regular', 'departure'] },
  },
  required: ['position', 'player', 'type'],
  additionalProperties: false,
} as const;

/**
 * Board state for serialization.
 */
export const BoardStateSchema = {
  $id: 'BoardState',
  type: 'object',
  properties: {
    type: { type: 'string', enum: ['square8', 'square19', 'hexagonal'] },
    size: { type: 'integer', minimum: 1 },
    stacks: {
      type: 'object',
      additionalProperties: { $ref: 'RingStack' },
      description: 'Map of position keys to stacks',
    },
    markers: {
      type: 'object',
      additionalProperties: { $ref: 'Marker' },
      description: 'Map of position keys to markers',
    },
    collapsedSpaces: {
      type: 'object',
      additionalProperties: { type: 'integer', minimum: 1 },
      description: 'Map of position keys to controlling player',
    },
    eliminatedRings: {
      type: 'object',
      additionalProperties: { type: 'integer', minimum: 0 },
      description: 'Map of player number to eliminated ring count',
    },
  },
  required: ['type', 'size', 'stacks', 'markers', 'collapsedSpaces', 'eliminatedRings'],
  additionalProperties: true,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Game State Schema
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Player state representation.
 */
export const PlayerStateSchema = {
  $id: 'PlayerState',
  type: 'object',
  properties: {
    playerNumber: { type: 'integer', minimum: 1 },
    ringsInHand: { type: 'integer', minimum: 0 },
    eliminatedRings: { type: 'integer', minimum: 0 },
    territorySpaces: { type: 'integer', minimum: 0 },
    isActive: { type: 'boolean' },
  },
  required: ['playerNumber', 'ringsInHand', 'eliminatedRings', 'territorySpaces'],
  additionalProperties: true,
} as const;

/**
 * Game phase enumeration.
 */
export const GamePhaseSchema = {
  $id: 'GamePhase',
  type: 'string',
  enum: [
    'ring_placement',
    'movement',
    'capture',
    'chain_capture',
    'line_processing',
    'territory_processing',
    'game_over',
  ],
} as const;

/**
 * Complete game state for serialization.
 */
export const GameStateSchema = {
  $id: 'GameState',
  type: 'object',
  properties: {
    gameId: { type: 'string' },
    board: { $ref: 'BoardState' },
    players: {
      type: 'array',
      items: { $ref: 'PlayerState' },
      minItems: 2,
    },
    currentPlayer: { type: 'integer', minimum: 1 },
    currentPhase: { $ref: 'GamePhase' },
    turnNumber: { type: 'integer', minimum: 1 },
    moveHistory: {
      type: 'array',
      items: { $ref: 'Move' },
    },
    gameStatus: { type: 'string', enum: ['active', 'completed', 'abandoned'] },
    victoryThreshold: { type: 'integer', minimum: 1 },
    territoryVictoryThreshold: { type: 'integer', minimum: 1 },
  },
  required: ['board', 'players', 'currentPlayer', 'currentPhase', 'turnNumber', 'gameStatus'],
  additionalProperties: true,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn Request/Response Schemas
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Request format for processTurn.
 */
export const ProcessTurnRequestSchema = {
  $id: 'ProcessTurnRequest',
  type: 'object',
  properties: {
    state: { $ref: 'GameState' },
    move: { $ref: 'Move' },
  },
  required: ['state', 'move'],
  additionalProperties: false,
} as const;

/**
 * Pending decision types.
 */
export const DecisionTypeSchema = {
  $id: 'DecisionType',
  type: 'string',
  enum: [
    'line_order',
    'line_reward',
    'region_order',
    'elimination_target',
    'capture_direction',
    'chain_capture',
  ],
} as const;

/**
 * Pending decision requiring player input.
 */
export const PendingDecisionSchema = {
  $id: 'PendingDecision',
  type: 'object',
  properties: {
    type: { $ref: 'DecisionType' },
    player: { type: 'integer', minimum: 1 },
    options: {
      type: 'array',
      items: { $ref: 'Move' },
      minItems: 1,
    },
    context: {
      type: 'object',
      properties: {
        description: { type: 'string' },
        relevantPositions: {
          type: 'array',
          items: { $ref: 'Position' },
        },
      },
      required: ['description'],
    },
  },
  required: ['type', 'player', 'options', 'context'],
  additionalProperties: false,
} as const;

/**
 * Victory reason enumeration.
 */
export const VictoryReasonSchema = {
  $id: 'VictoryReason',
  type: 'string',
  enum: [
    'ring_elimination',
    'territory_control',
    'last_player_standing',
    'stalemate_resolution',
    'resignation',
  ],
} as const;

/**
 * Victory state information.
 */
export const VictoryStateSchema = {
  $id: 'VictoryState',
  type: 'object',
  properties: {
    isGameOver: { type: 'boolean' },
    winner: { type: ['integer', 'null'], minimum: 1 },
    reason: { $ref: 'VictoryReason' },
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          player: { type: 'integer', minimum: 1 },
          eliminatedRings: { type: 'integer', minimum: 0 },
          territorySpaces: { type: 'integer', minimum: 0 },
          ringsOnBoard: { type: 'integer', minimum: 0 },
          ringsInHand: { type: 'integer', minimum: 0 },
          markerCount: { type: 'integer', minimum: 0 },
          isEliminated: { type: 'boolean' },
        },
        required: ['player', 'eliminatedRings', 'territorySpaces'],
      },
    },
  },
  required: ['isGameOver'],
  additionalProperties: true,
} as const;

/**
 * Processing metadata for debugging.
 */
export const ProcessingMetadataSchema = {
  $id: 'ProcessingMetadata',
  type: 'object',
  properties: {
    processedMove: { $ref: 'Move' },
    phasesTraversed: {
      type: 'array',
      items: { type: 'string' },
    },
    linesDetected: { type: 'integer', minimum: 0 },
    regionsProcessed: { type: 'integer', minimum: 0 },
    durationMs: { type: 'number', minimum: 0 },
    sInvariantBefore: { type: 'integer', minimum: 0 },
    sInvariantAfter: { type: 'integer', minimum: 0 },
  },
  required: ['processedMove', 'phasesTraversed'],
  additionalProperties: true,
} as const;

/**
 * Response format for processTurn.
 */
export const ProcessTurnResponseSchema = {
  $id: 'ProcessTurnResponse',
  type: 'object',
  properties: {
    nextState: { $ref: 'GameState' },
    status: { type: 'string', enum: ['complete', 'awaiting_decision'] },
    pendingDecision: { $ref: 'PendingDecision' },
    victoryResult: { $ref: 'VictoryState' },
    metadata: { $ref: 'ProcessingMetadata' },
  },
  required: ['nextState', 'status', 'metadata'],
  additionalProperties: false,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Test Vector Schema
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Test vector for parity testing.
 */
export const TestVectorSchema = {
  $id: 'TestVector',
  type: 'object',
  properties: {
    id: { type: 'string', description: 'Unique test vector identifier' },
    description: { type: 'string' },
    category: {
      type: 'string',
      enum: [
        'placement',
        'movement',
        'capture',
        'chain_capture',
        'line_processing',
        'territory_processing',
        'victory',
        'edge_case',
      ],
    },
    input: { $ref: 'ProcessTurnRequest' },
    expectedOutput: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['complete', 'awaiting_decision'] },
        // Key assertions to check
        assertions: {
          type: 'object',
          properties: {
            currentPlayer: { type: 'integer' },
            currentPhase: { $ref: 'GamePhase' },
            gameStatus: { type: 'string' },
            stackCount: { type: 'integer' },
            markerCount: { type: 'integer' },
            collapsedCount: { type: 'integer' },
            sInvariantDelta: { type: 'integer' },
            victoryWinner: { type: ['integer', 'null'] },
          },
        },
      },
      required: ['status'],
    },
    tags: {
      type: 'array',
      items: { type: 'string' },
      description: 'Tags for filtering (e.g., "regression", "parity", "edge-case")',
    },
    source: {
      type: 'string',
      description: 'Origin of test vector (e.g., "manual", "recorded", "generated")',
    },
    createdAt: { type: 'string', format: 'date-time' },
  },
  required: ['id', 'category', 'input', 'expectedOutput'],
  additionalProperties: true,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// All Schemas Export
// ═══════════════════════════════════════════════════════════════════════════

/**
 * All schemas bundled for JSON Schema validator registration.
 */
export const AllSchemas = {
  Position: PositionSchema,
  MoveType: MoveTypeSchema,
  Move: MoveSchema,
  RingStack: RingStackSchema,
  Marker: MarkerSchema,
  BoardState: BoardStateSchema,
  PlayerState: PlayerStateSchema,
  GamePhase: GamePhaseSchema,
  GameState: GameStateSchema,
  ProcessTurnRequest: ProcessTurnRequestSchema,
  DecisionType: DecisionTypeSchema,
  PendingDecision: PendingDecisionSchema,
  VictoryReason: VictoryReasonSchema,
  VictoryState: VictoryStateSchema,
  ProcessingMetadata: ProcessingMetadataSchema,
  ProcessTurnResponse: ProcessTurnResponseSchema,
  TestVector: TestVectorSchema,
} as const;

/**
 * JSON Schema bundle for export (e.g., for Python consumption).
 */
export function exportSchemaBundle(): object {
  return {
    $schema: 'https://json-schema.org/draft/2020-12/schema',
    $id: 'RingRiftEngineContracts',
    title: 'RingRift Engine Contract Schemas',
    description: 'JSON Schema definitions for TypeScript/Python engine parity',
    version: '1.0.0',
    definitions: AllSchemas,
  };
}
