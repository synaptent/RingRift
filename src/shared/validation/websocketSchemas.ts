import { z } from 'zod';
import { MoveSchema } from './schemas';

/**
 * Zod schemas for incoming WebSocket payloads.
 *
 * These schemas mirror the current contracts used by WebSocketServer and
 * GameSession. They are intentionally narrower than some internal domain
 * types so we can evolve backend representations without breaking clients.
 */

// --- Core game room events ---

export const JoinGamePayloadSchema = z.object({
  gameId: z.string().min(1),
});

export type JoinGamePayload = z.infer<typeof JoinGamePayloadSchema>;

export const LeaveGamePayloadSchema = z.object({
  gameId: z.string().min(1),
});

export type LeaveGamePayload = z.infer<typeof LeaveGamePayloadSchema>;

// --- Move submission events ---

export const PlayerMovePayloadSchema = z.object({
  gameId: z.string().min(1),
  move: MoveSchema,
});

export type PlayerMovePayload = z.infer<typeof PlayerMovePayloadSchema>;

export const PlayerMoveByIdPayloadSchema = z.object({
  gameId: z.string().min(1),
  moveId: z.string().min(1),
});

export type PlayerMoveByIdPayload = z.infer<typeof PlayerMoveByIdPayloadSchema>;

// --- Chat events ---

export const ChatMessagePayloadSchema = z.object({
  gameId: z.string().min(1),
  // The existing chat handler expects a simple { gameId, text } payload and
  // broadcasts the same text field back out. We intentionally avoid
  // UUID-only or HTTP ChatMessageSchema constraints here so tests and
  // legacy clients that use synthetic ids continue to work.
  text: z
    .string()
    .min(1, 'Message cannot be empty')
    .max(500, 'Message must be at most 500 characters')
    .trim(),
});

export type ChatMessagePayload = z.infer<typeof ChatMessagePayloadSchema>;

// --- Player choice system events ---

/**
 * PlayerChoiceResponse is structurally validated enough to protect the
 * interaction handler, but we deliberately keep selectedOption as
 * z.unknown() so that per-choice-type option shapes remain the single
 * source of truth on the engine side. WebSocketInteractionHandler performs
 * the semantic check that selectedOption matches one of choice.options.
 */
export const PlayerChoiceResponsePayloadSchema = z.object({
  choiceId: z.string().min(1),
  playerNumber: z.number().int().min(1),
  choiceType: z
    .enum([
      'line_order',
      'line_reward_option',
      'ring_elimination',
      'region_order',
      'capture_direction',
    ])
    .optional(),
  selectedOption: z.unknown(),
});

export type PlayerChoiceResponsePayload = z.infer<typeof PlayerChoiceResponsePayloadSchema>;

// --- Rematch system events ---

/**
 * Client-to-server payload for requesting a rematch after a game ends.
 */
export const RematchRequestPayloadSchema = z.object({
  gameId: z.string().min(1),
});

export type RematchRequestPayload = z.infer<typeof RematchRequestPayloadSchema>;

/**
 * Client-to-server payload for responding to a rematch request.
 */
export const RematchResponsePayloadSchema = z.object({
  requestId: z.string().min(1),
  accept: z.boolean(),
});

export type RematchResponsePayload = z.infer<typeof RematchResponsePayloadSchema>;

// --- Matchmaking events ---

/**
 * Valid board types for matchmaking preferences.
 */
const BoardTypeSchema = z.enum(['square8', 'square19', 'hex8', 'hexagonal']);

/**
 * Numeric range schema for matchmaking preferences.
 */
const NumericRangeSchema = z
  .object({
    min: z.number().int().nonnegative(),
    max: z.number().int().nonnegative(),
  })
  .refine((data) => data.min <= data.max, {
    message: 'min must be less than or equal to max',
  });

/**
 * Matchmaking preferences for finding opponents.
 */
const MatchmakingPreferencesSchema = z.object({
  boardType: BoardTypeSchema,
  ratingRange: NumericRangeSchema,
  timeControl: NumericRangeSchema,
});

/**
 * Client-to-server payload for joining the matchmaking queue.
 */
export const MatchmakingJoinPayloadSchema = z.object({
  preferences: MatchmakingPreferencesSchema,
});

export type MatchmakingJoinPayload = z.infer<typeof MatchmakingJoinPayloadSchema>;

// --- Diagnostic / load-testing events ---

/**
 * Lightweight ping payload used by WebSocket load tests. This is intentionally
 * generic and non-game-specific so it can be exercised at high volume without
 * touching the rules engine or database.
 */
export const DiagnosticPingPayloadSchema = z.object({
  /**
   * Client-side timestamp in milliseconds since epoch. Used by load tests to
   * compute round-trip latency without relying on server clocks.
   */
  timestamp: z.number().int().nonnegative(),
  /**
   * Optional virtual-user identifier or other opaque tag attached by the
   * caller. This is propagated unchanged in the diagnostic:pong response.
   */
  vu: z.union([z.number().int().nonnegative(), z.string()]).optional(),
  /**
   * Optional monotonically-increasing sequence number for correlating
   * individual ping/pong pairs. Also echoed back verbatim.
   */
  sequence: z.number().int().nonnegative().optional(),
});

export type DiagnosticPingPayload = z.infer<typeof DiagnosticPingPayloadSchema>;

// --- Server → client metadata schemas (game_state diff summaries) ---

/**
 * Reason why a pending decision was auto-resolved by the system.
 *
 * This mirrors DecisionAutoResolveReason in src/shared/types/websocket.ts.
 */
export const DecisionAutoResolveReasonSchema = z.enum(['timeout', 'disconnected', 'fallback']);

/**
 * High-level semantic grouping of a decision, aligned with ChoiceViewModels.
 *
 * This mirrors DecisionChoiceKind in src/shared/types/websocket.ts.
 */
export const DecisionChoiceKindSchema = z.enum([
  'line_order',
  'line_reward',
  'ring_elimination',
  'territory_region_order',
  'capture_direction',
  'other',
]);

/**
 * Zod schema for the DecisionAutoResolvedMeta structure attached to
 * GameStateUpdateMessage.data.meta.diffSummary.decisionAutoResolved.
 */
export const DecisionAutoResolvedMetaSchema = z.object({
  choiceType: z.enum([
    'line_order',
    'line_reward_option',
    'ring_elimination',
    'region_order',
    'capture_direction',
  ]),
  choiceKind: DecisionChoiceKindSchema,
  actingPlayerNumber: z.number().int().min(1),
  resolvedMoveId: z.string().min(1).optional(),
  resolvedOptionIndex: z.number().int().min(0).optional(),
  resolvedOptionKey: z.string().min(1).optional(),
  reason: DecisionAutoResolveReasonSchema,
});

export type DecisionAutoResolvedMetaPayload = z.infer<typeof DecisionAutoResolvedMetaSchema>;

/**
 * Schema for the optional meta block on GameStateUpdateMessage.data.
 *
 * This is intentionally focused on diffSummary so that tests can validate
 * the presence and shape of decisionAutoResolved metadata without
 * constraining the full GameState representation used on the wire.
 */
export const GameStateUpdateMetaSchema = z.object({
  diffSummary: z
    .object({
      decisionAutoResolved: DecisionAutoResolvedMetaSchema.optional(),
    })
    .optional(),
});

// --- Event name &#8594; schema mapping used by WebSocketServer ---

export const WebSocketPayloadSchemas = {
  join_game: JoinGamePayloadSchema,
  leave_game: LeaveGamePayloadSchema,
  player_move: PlayerMovePayloadSchema,
  player_move_by_id: PlayerMoveByIdPayloadSchema,
  chat_message: ChatMessagePayloadSchema,
  player_choice_response: PlayerChoiceResponsePayloadSchema,
  rematch_request: RematchRequestPayloadSchema,
  rematch_respond: RematchResponsePayloadSchema,
  'matchmaking:join': MatchmakingJoinPayloadSchema,
  'diagnostic:ping': DiagnosticPingPayloadSchema,
} as const;

export type WebSocketEventName = keyof typeof WebSocketPayloadSchemas;
