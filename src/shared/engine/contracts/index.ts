/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Contracts Module - Cross-Engine Parity Definitions
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module provides:
 * 1. JSON Schema definitions for engine API contracts
 * 2. Serialization utilities for cross-language communication
 * 3. Test vector helpers for parity testing
 *
 * Used by:
 * - Python AI service for contract-based testing
 * - Test infrastructure for parity validation
 * - Future: runtime schema validation
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
