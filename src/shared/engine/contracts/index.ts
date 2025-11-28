/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Contracts Module - Cross-Engine Parity Definitions
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module provides:
 * 1. JSON Schema definitions for engine API contracts
 * 2. Zod-based runtime validation for boundary checking
 * 3. Serialization utilities for cross-language communication
 * 4. Test vector helpers for parity testing
 *
 * Used by:
 * - Python AI service for contract-based testing
 * - Test infrastructure for parity validation
 * - Runtime schema validation at boundary points
 */

// JSON Schema definitions
export {
  // Core schemas
  PositionSchema,
  MoveTypeSchema,
  MoveSchema,
  RingStackSchema,
  MarkerSchema,
  BoardStateSchema,
  PlayerStateSchema,
  GamePhaseSchema,
  GameStateSchema,

  // API schemas
  ProcessTurnRequestSchema,
  ProcessTurnResponseSchema,
  DecisionTypeSchema,
  PendingDecisionSchema,
  VictoryReasonSchema,
  VictoryStateSchema,
  ProcessingMetadataSchema,

  // Test schemas
  TestVectorSchema,

  // Bundle exports
  AllSchemas,
  exportSchemaBundle,
} from './schemas';

// Serialization utilities
export {
  // Types
  type SerializedBoardState,
  type SerializedStack,
  type SerializedMarker,
  type SerializedGameState,

  // Board serialization
  serializeBoardState,
  deserializeBoardState,

  // Game state serialization
  serializeGameState,
  deserializeGameState,
  gameStateToJson,
  jsonToGameState,

  // Test vector helpers
  createTestVector,
  computeStateDiff,
} from './serialization';

// Test vector generator
export {
  type TestVectorCategory,
  type ContractTestVector,
  type TestVectorAssertions,
  inferCategory,
  generateVectorId,
  createContractTestVector,
  createTestVectorsFromTrace,
  validateAgainstAssertions,
  exportVectorBundle,
  importVectorBundle,
} from './testVectorGenerator';

// Runtime validators (Zod-based)
export {
  // Zod schemas
  ZodPositionSchema,
  ZodMoveTypeSchema,
  ZodMoveSchema,
  ZodRingStackSchema,
  ZodMarkerSchema,
  ZodLineInfoSchema,
  ZodTerritorySchema,
  ZodPlayerStateSchema,
  ZodBoardTypeSchema,
  ZodSerializedBoardStateSchema,
  ZodGamePhaseSchema,
  ZodGameStatusSchema,
  ZodSerializedGameStateSchema,
  ZodProcessTurnRequestSchema,
  ZodDecisionTypeSchema,
  ZodPendingDecisionSchema,
  ZodVictoryReasonSchema,
  ZodVictoryStateSchema,
  ZodProcessingMetadataSchema,
  ZodProcessTurnResponseSchema,
  ZodMoveResultSchema,
  ZodSchemas,

  // Zod-inferred types
  type ZodPosition,
  type ZodMoveType,
  type ZodMove,
  type ZodRingStack,
  type ZodMarker,
  type ZodLineInfo,
  type ZodTerritory,
  type ZodPlayerState,
  type ZodSerializedBoardState,
  type ZodGamePhase,
  type ZodGameStatus,
  type ZodSerializedGameState,
  type ZodProcessTurnRequest,
  type ZodDecisionType,
  type ZodPendingDecision,
  type ZodVictoryReason,
  type ZodVictoryState,
  type ZodProcessingMetadata,
  type ZodProcessTurnResponse,
  type ZodMoveResult,

  // Validation functions (safe - returns result object)
  validatePosition,
  validateMove,
  validateRingStack,
  validateMarker,
  validatePlayerState,
  validateSerializedBoardState,
  validateSerializedGameState,
  validateProcessTurnRequest,
  validateProcessTurnResponse,
  validateVictoryState,
  validateMoveResult,

  // Parse functions (strict - throws on error)
  parsePosition,
  parseMove,
  parseRingStack,
  parseMarker,
  parsePlayerState,
  parseSerializedBoardState,
  parseSerializedGameState,
  parseProcessTurnRequest,
  parseProcessTurnResponse,
  parseVictoryState,
  parseMoveResult,

  // Utilities
  formatZodError,
  createValidator,
  type ValidationResult,
} from './validators';
