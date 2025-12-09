/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Runtime Validators for Engine Contracts
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This file provides Zod-based runtime validation for engine contracts.
 * Used at boundary points (client-server, server-AI service) to ensure
 * data integrity and catch malformed states early.
 *
 * Each schema has two validation functions:
 * - validate*: Returns { success: true, data } | { success: false, error }
 * - parse*: Throws on invalid data, returns typed data on success
 */

import { z } from 'zod';

// ═══════════════════════════════════════════════════════════════════════════
// Position Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Zod schema for Position.
 * For square boards: {x, y}
 * For hexagonal boards: {x, y, z} using cube coordinates
 */
export const ZodPositionSchema = z.object({
  x: z.number().int(),
  y: z.number().int(),
  z: z.number().int().optional(),
});

export type ZodPosition = z.infer<typeof ZodPositionSchema>;

export function validatePosition(
  data: unknown
): { success: true; data: ZodPosition } | { success: false; error: string } {
  const result = ZodPositionSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parsePosition(data: unknown): ZodPosition {
  return ZodPositionSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Move Type Schema
// ═══════════════════════════════════════════════════════════════════════════

export const ZodMoveTypeSchema = z.enum([
  'place_ring',
  'move_ring',
  'build_stack',
  'move_stack',
  'skip_placement',
  'overtaking_capture',
  'continue_capture_segment',
  'recovery_slide', // RR-CANON-R110–R115: marker recovery action
  'process_line',
  'choose_line_reward',
  'process_territory_region',
  'eliminate_rings_from_stack',
  'forced_elimination',
  'no_territory_action',
  'no_line_action',
  'line_formation',
  'territory_claim',
]);

export type ZodMoveType = z.infer<typeof ZodMoveTypeSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// Ring Stack Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodRingStackSchema = z.object({
  position: ZodPositionSchema,
  rings: z.array(z.number().int().min(1)),
  stackHeight: z.number().int().min(0),
  capHeight: z.number().int().min(0),
  controllingPlayer: z.number().int().min(1),
});

export type ZodRingStack = z.infer<typeof ZodRingStackSchema>;

export function validateRingStack(
  data: unknown
): { success: true; data: ZodRingStack } | { success: false; error: string } {
  const result = ZodRingStackSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseRingStack(data: unknown): ZodRingStack {
  return ZodRingStackSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Marker Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodMarkerSchema = z.object({
  position: ZodPositionSchema,
  player: z.number().int().min(1),
  type: z.enum(['regular', 'collapsed', 'departure']),
});

export type ZodMarker = z.infer<typeof ZodMarkerSchema>;

export function validateMarker(
  data: unknown
): { success: true; data: ZodMarker } | { success: false; error: string } {
  const result = ZodMarkerSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseMarker(data: unknown): ZodMarker {
  return ZodMarkerSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Line Info Schema
// ═══════════════════════════════════════════════════════════════════════════

export const ZodLineInfoSchema = z.object({
  positions: z.array(ZodPositionSchema),
  player: z.number().int().min(1),
  length: z.number().int().min(1),
  direction: ZodPositionSchema,
});

export type ZodLineInfo = z.infer<typeof ZodLineInfoSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// Territory Schema
// ═══════════════════════════════════════════════════════════════════════════

export const ZodTerritorySchema = z.object({
  spaces: z.array(ZodPositionSchema),
  controllingPlayer: z.number().int().min(1),
  isDisconnected: z.boolean(),
});

export type ZodTerritory = z.infer<typeof ZodTerritorySchema>;

// ═══════════════════════════════════════════════════════════════════════════
// Move Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodMoveSchema = z.object({
  id: z.string(),
  type: ZodMoveTypeSchema,
  player: z.number().int().min(1),
  from: ZodPositionSchema.optional(),
  to: ZodPositionSchema,

  // Ring placement specific
  placedOnStack: z.boolean().optional(),
  placementCount: z.number().int().min(1).optional(),

  // Movement specific
  stackMoved: ZodRingStackSchema.optional(),
  minimumDistance: z.number().int().min(0).optional(),
  actualDistance: z.number().int().min(0).optional(),
  markerLeft: ZodPositionSchema.optional(),
  buildAmount: z.number().int().min(1).optional(),

  // Capture specific
  captureType: z.enum(['overtaking', 'elimination']).optional(),
  captureTarget: ZodPositionSchema.optional(),
  capturedStacks: z.array(ZodRingStackSchema).optional(),
  captureChain: z.array(ZodPositionSchema).optional(),
  overtakenRings: z.array(z.number().int().min(1)).optional(),

  // Line formation specific
  formedLines: z.array(ZodLineInfoSchema).optional(),
  collapsedMarkers: z.array(ZodPositionSchema).optional(),

  // Territory specific
  claimedTerritory: z.array(ZodTerritorySchema).optional(),
  disconnectedRegions: z.array(ZodTerritorySchema).optional(),
  eliminatedRings: z
    .array(
      z.object({
        player: z.number().int().min(1),
        count: z.number().int().min(0),
      })
    )
    .optional(),
  eliminationFromStack: z
    .object({
      position: ZodPositionSchema,
      capHeight: z.number().int().min(0),
      totalHeight: z.number().int().min(0),
    })
    .optional(),

  // Metadata
  timestamp: z.union([z.date(), z.string()]),
  thinkTime: z.number().min(0),
  moveNumber: z.number().int().min(1),
});

export type ZodMove = z.infer<typeof ZodMoveSchema>;

export function validateMove(
  data: unknown
): { success: true; data: ZodMove } | { success: false; error: string } {
  const result = ZodMoveSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseMove(data: unknown): ZodMove {
  return ZodMoveSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Player State Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodPlayerStateSchema = z.object({
  playerNumber: z.number().int().min(1),
  ringsInHand: z.number().int().min(0),
  eliminatedRings: z.number().int().min(0),
  territorySpaces: z.number().int().min(0),
  isActive: z.boolean().optional(),
});

export type ZodPlayerState = z.infer<typeof ZodPlayerStateSchema>;

export function validatePlayerState(
  data: unknown
): { success: true; data: ZodPlayerState } | { success: false; error: string } {
  const result = ZodPlayerStateSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parsePlayerState(data: unknown): ZodPlayerState {
  return ZodPlayerStateSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Board State Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodBoardTypeSchema = z.enum(['square8', 'square19', 'hexagonal']);

export const ZodSerializedBoardStateSchema = z.object({
  type: ZodBoardTypeSchema,
  size: z.number().int().min(1),
  stacks: z.record(z.string(), ZodRingStackSchema),
  markers: z.record(z.string(), ZodMarkerSchema),
  collapsedSpaces: z.record(z.string(), z.number().int().min(1)),
  // Keys are already JSON object keys (strings); coercion is unnecessary and
  // incompatible with Zod 4's stricter record key constraints, so we accept
  // plain string keys here.
  eliminatedRings: z.record(z.string(), z.number().int().min(0)),
  formedLines: z.array(z.unknown()).optional(),
});

export type ZodSerializedBoardState = z.infer<typeof ZodSerializedBoardStateSchema>;

export function validateSerializedBoardState(
  data: unknown
): { success: true; data: ZodSerializedBoardState } | { success: false; error: string } {
  const result = ZodSerializedBoardStateSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseSerializedBoardState(data: unknown): ZodSerializedBoardState {
  return ZodSerializedBoardStateSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Game Phase & Status Schemas
// ═══════════════════════════════════════════════════════════════════════════

export const ZodGamePhaseSchema = z.enum([
  'ring_placement',
  'movement',
  'capture',
  'chain_capture',
  'line_processing',
  'territory_processing',
  'forced_elimination',
]);

export type ZodGamePhase = z.infer<typeof ZodGamePhaseSchema>;

export const ZodGameStatusSchema = z.enum([
  'waiting',
  'active',
  'finished',
  'paused',
  'abandoned',
  'completed',
]);

export type ZodGameStatus = z.infer<typeof ZodGameStatusSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// Serialized Game State Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodSerializedGameStateSchema = z.object({
  gameId: z.string().optional(),
  board: ZodSerializedBoardStateSchema,
  players: z.array(ZodPlayerStateSchema).min(2),
  currentPlayer: z.number().int().min(1),
  currentPhase: ZodGamePhaseSchema,
  turnNumber: z.number().int().min(1),
  moveHistory: z.array(ZodMoveSchema),
  gameStatus: ZodGameStatusSchema,
  victoryThreshold: z.number().int().min(1).optional(),
  territoryVictoryThreshold: z.number().int().min(1).optional(),
  totalRingsEliminated: z.number().int().min(0).optional(),
});

export type ZodSerializedGameState = z.infer<typeof ZodSerializedGameStateSchema>;

export function validateSerializedGameState(
  data: unknown
): { success: true; data: ZodSerializedGameState } | { success: false; error: string } {
  const result = ZodSerializedGameStateSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseSerializedGameState(data: unknown): ZodSerializedGameState {
  return ZodSerializedGameStateSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn Request Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodProcessTurnRequestSchema = z.object({
  state: ZodSerializedGameStateSchema,
  move: ZodMoveSchema,
});

export type ZodProcessTurnRequest = z.infer<typeof ZodProcessTurnRequestSchema>;

export function validateProcessTurnRequest(
  data: unknown
): { success: true; data: ZodProcessTurnRequest } | { success: false; error: string } {
  const result = ZodProcessTurnRequestSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseProcessTurnRequest(data: unknown): ZodProcessTurnRequest {
  return ZodProcessTurnRequestSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Decision Type & Pending Decision Schemas
// ═══════════════════════════════════════════════════════════════════════════

export const ZodDecisionTypeSchema = z.enum([
  'line_order',
  'line_reward',
  'region_order',
  'elimination_target',
  'capture_direction',
  'chain_capture',
]);

export type ZodDecisionType = z.infer<typeof ZodDecisionTypeSchema>;

export const ZodPendingDecisionSchema = z.object({
  type: ZodDecisionTypeSchema,
  player: z.number().int().min(1),
  options: z.array(ZodMoveSchema).min(1),
  context: z.object({
    description: z.string(),
    relevantPositions: z.array(ZodPositionSchema).optional(),
  }),
});

export type ZodPendingDecision = z.infer<typeof ZodPendingDecisionSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// Victory Schemas
// ═══════════════════════════════════════════════════════════════════════════

export const ZodVictoryReasonSchema = z.enum([
  'ring_elimination',
  'territory_control',
  'last_player_standing',
  'stalemate_resolution',
  'resignation',
]);

export type ZodVictoryReason = z.infer<typeof ZodVictoryReasonSchema>;

export const ZodVictoryStateSchema = z.object({
  isGameOver: z.boolean(),
  winner: z.number().int().min(1).nullable().optional(),
  reason: ZodVictoryReasonSchema.optional(),
  scores: z
    .array(
      z.object({
        player: z.number().int().min(1),
        eliminatedRings: z.number().int().min(0),
        territorySpaces: z.number().int().min(0),
        ringsOnBoard: z.number().int().min(0).optional(),
        ringsInHand: z.number().int().min(0).optional(),
        markerCount: z.number().int().min(0).optional(),
        isEliminated: z.boolean().optional(),
      })
    )
    .optional(),
});

export type ZodVictoryState = z.infer<typeof ZodVictoryStateSchema>;

export function validateVictoryState(
  data: unknown
): { success: true; data: ZodVictoryState } | { success: false; error: string } {
  const result = ZodVictoryStateSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseVictoryState(data: unknown): ZodVictoryState {
  return ZodVictoryStateSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Processing Metadata Schema
// ═══════════════════════════════════════════════════════════════════════════

export const ZodProcessingMetadataSchema = z.object({
  processedMove: ZodMoveSchema,
  phasesTraversed: z.array(z.string()),
  linesDetected: z.number().int().min(0).optional(),
  regionsProcessed: z.number().int().min(0).optional(),
  durationMs: z.number().min(0).optional(),
  sInvariantBefore: z.number().int().min(0).optional(),
  sInvariantAfter: z.number().int().min(0).optional(),
});

export type ZodProcessingMetadata = z.infer<typeof ZodProcessingMetadataSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn Response Schema & Validators
// ═══════════════════════════════════════════════════════════════════════════

export const ZodProcessTurnResponseSchema = z.object({
  nextState: ZodSerializedGameStateSchema,
  status: z.enum(['complete', 'awaiting_decision']),
  pendingDecision: ZodPendingDecisionSchema.optional(),
  victoryResult: ZodVictoryStateSchema.optional(),
  metadata: ZodProcessingMetadataSchema,
});

export type ZodProcessTurnResponse = z.infer<typeof ZodProcessTurnResponseSchema>;

export function validateProcessTurnResponse(
  data: unknown
): { success: true; data: ZodProcessTurnResponse } | { success: false; error: string } {
  const result = ZodProcessTurnResponseSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseProcessTurnResponse(data: unknown): ZodProcessTurnResponse {
  return ZodProcessTurnResponseSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Move Result Schema & Validators (Simplified response for move execution)
// ═══════════════════════════════════════════════════════════════════════════

export const ZodMoveResultSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),
  newState: ZodSerializedGameStateSchema.optional(),
  processedMove: ZodMoveSchema.optional(),
  awaitingDecision: z.boolean().optional(),
  pendingDecision: ZodPendingDecisionSchema.optional(),
  victoryResult: ZodVictoryStateSchema.optional(),
});

export type ZodMoveResult = z.infer<typeof ZodMoveResultSchema>;

export function validateMoveResult(
  data: unknown
): { success: true; data: ZodMoveResult } | { success: false; error: string } {
  const result = ZodMoveResultSchema.safeParse(data);
  if (!result.success) {
    return { success: false, error: formatZodError(result.error) };
  }
  return { success: true, data: result.data };
}

export function parseMoveResult(data: unknown): ZodMoveResult {
  return ZodMoveResultSchema.parse(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Error Formatting Helper
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format a Zod error into a human-readable string.
 */
export function formatZodError(error: z.ZodError | undefined | null): string {
  const zodError = error as z.ZodError | undefined;
  // Zod 4 exposes `issues`; older versions used `errors`. Support both defensively.
  const issues = zodError?.issues ?? (zodError as { errors?: z.ZodIssue[] } | undefined)?.errors;

  if (!issues || !Array.isArray(issues) || issues.length === 0) {
    // Fallback for unexpected inputs; preserve basic error information if present.
    if (error instanceof Error && error.message) {
      return error.message;
    }
    return 'Invalid data';
  }

  return issues
    .map((issue) => {
      const zodIssue = issue as z.ZodIssue;

      const rawPath = Array.isArray(zodIssue.path) ? zodIssue.path : [];
      const normalizedPath = rawPath.map((seg: unknown) =>
        typeof seg === 'string' || typeof seg === 'number' ? seg : String(seg)
      );
      const pathPrefix = normalizedPath.length > 0 ? `${normalizedPath.join('.')}: ` : '';

      const message =
        typeof zodIssue.message === 'string' && zodIssue.message.length > 0
          ? zodIssue.message
          : 'Invalid data';

      return `${pathPrefix}${message}`;
    })
    .join('; ');
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Result Type
// ═══════════════════════════════════════════════════════════════════════════

export type ValidationResult<T> = { success: true; data: T } | { success: false; error: string };

// ═══════════════════════════════════════════════════════════════════════════
// Generic Validation Wrapper
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a validation function for any Zod schema.
 * Returns a tuple of [validate, parse] functions.
 */
export function createValidator<T extends z.ZodType>(
  schema: T
): [(data: unknown) => ValidationResult<z.infer<T>>, (data: unknown) => z.infer<T>] {
  const validate = (data: unknown): ValidationResult<z.infer<T>> => {
    const result = schema.safeParse(data);
    if (!result.success) {
      return { success: false, error: formatZodError(result.error) };
    }
    return { success: true, data: result.data };
  };

  const parse = (data: unknown): z.infer<T> => {
    return schema.parse(data);
  };

  return [validate, parse];
}

// ═══════════════════════════════════════════════════════════════════════════
// All Zod Schemas Export (for registration/reuse)
// ═══════════════════════════════════════════════════════════════════════════

export const ZodSchemas = {
  Position: ZodPositionSchema,
  MoveType: ZodMoveTypeSchema,
  Move: ZodMoveSchema,
  RingStack: ZodRingStackSchema,
  Marker: ZodMarkerSchema,
  LineInfo: ZodLineInfoSchema,
  Territory: ZodTerritorySchema,
  PlayerState: ZodPlayerStateSchema,
  BoardType: ZodBoardTypeSchema,
  SerializedBoardState: ZodSerializedBoardStateSchema,
  GamePhase: ZodGamePhaseSchema,
  GameStatus: ZodGameStatusSchema,
  SerializedGameState: ZodSerializedGameStateSchema,
  ProcessTurnRequest: ZodProcessTurnRequestSchema,
  DecisionType: ZodDecisionTypeSchema,
  PendingDecision: ZodPendingDecisionSchema,
  VictoryReason: ZodVictoryReasonSchema,
  VictoryState: ZodVictoryStateSchema,
  ProcessingMetadata: ZodProcessingMetadataSchema,
  ProcessTurnResponse: ZodProcessTurnResponseSchema,
  MoveResult: ZodMoveResultSchema,
} as const;
