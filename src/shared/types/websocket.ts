import type {
  BoardType,
  Game,
  GameResult,
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceType,
} from './game';
import type {
  JoinGamePayload,
  LeaveGamePayload,
  PlayerMovePayload,
  PlayerMoveByIdPayload,
  ChatMessagePayload,
  PlayerChoiceResponsePayload,
} from '../validation/websocketSchemas';

/**
 * Player matchmaking preferences for finding opponents.
 */
export interface MatchmakingPreferences {
  boardType: BoardType;
  ratingRange: { min: number; max: number };
  timeControl: { min: number; max: number };
}

/**
 * Current matchmaking status for a player in the queue.
 */
export interface MatchmakingStatus {
  inQueue: boolean;
  estimatedWaitTime: number;
  queuePosition: number;
  searchCriteria: MatchmakingPreferences;
}

/**
 * Payload for match-found events.
 */
export interface MatchFoundPayload {
  gameId: string;
}

/**
 * Error codes used in structured WebSocket error payloads.
 *
 * These are emitted by WebSocketServer and consumed by the client
 * (see GameContext). They should remain stable over time so that
 * clients can reliably distinguish categories of failures.
 */
export type WebSocketErrorCode =
  | 'INVALID_PAYLOAD'
  | 'GAME_NOT_FOUND'
  | 'ACCESS_DENIED'
  | 'RATE_LIMITED'
  | 'MOVE_REJECTED'
  | 'CHOICE_REJECTED'
  | 'DECISION_PHASE_TIMEOUT'
  | 'INTERNAL_ERROR';

export interface WebSocketErrorPayload {
  type: 'error';
  code: WebSocketErrorCode;
  /**
   * Name of the event that triggered this error, when known
   * (e.g. 'join_game', 'player_move').
   */
  event?: string;
  message: string;
}

/**
 * Canonical payload for server → client chat messages.
 *
 * This is distinct from the inbound ChatMessagePayload, which only
 * contains { gameId, text }.
 */
export interface ChatMessageServerPayload {
  sender: string;
  text: string;
  timestamp: string;
}

/**
 * Reason why a pending decision was auto-resolved by the system.
 *
 * This is intentionally coarse-grained so that clients and logs can
 * distinguish user-driven decisions from system fallbacks.
 */
export type DecisionAutoResolveReason = 'timeout' | 'disconnected' | 'fallback';

/**
 * High-level semantic grouping of a decision, aligned with ChoiceViewModels.
 *
 * This mirrors the ChoiceKind union in src/client/adapters/choiceViewModels.ts
 * without introducing a dependency on client code.
 */
export type DecisionChoiceKind =
  | 'line_order'
  | 'line_reward'
  | 'ring_elimination'
  | 'territory_region_order'
  | 'capture_direction'
  | 'other';

/**
 * Summary of a decision that was auto-resolved by the server (e.g. due to
 * timeout). Attached to GameStateUpdateMessage.data.meta.diffSummary so
 * clients and logs can render user-vs-system decision UX.
 */
export interface DecisionAutoResolvedMeta {
  /** Underlying low-level discriminant from the originating PlayerChoice. */
  choiceType: PlayerChoiceType;
  /** High-level semantic grouping derived from ChoiceViewModels. */
  choiceKind: DecisionChoiceKind;
  /** Numeric player index whose decision was auto-resolved. */
  actingPlayerNumber: number;
  /**
   * When the decision corresponds directly to a canonical Move, this is the
   * stable Move.id that was applied.
   */
  resolvedMoveId?: string;
  /** Optional index into the original PlayerChoice.options array. */
  resolvedOptionIndex?: number;
  /** Optional key for stringly-typed options (e.g. line reward variants). */
  resolvedOptionKey?: string;
  /** Coarse-grained reason why this decision was auto-resolved. */
  reason: DecisionAutoResolveReason;
}

/**
 * Canonical payload for `game_state` events.
 *
 * Emitted:
 * - when a player joins a game (initial snapshot)
 * - after each successful move while the game is active
 *
 * Consumed by:
 * - GameContext on the client
 */
export interface GameStateUpdateMessage {
  type: 'game_update';
  data: {
    gameId: string;
    gameState: GameState;
    /**
     * Legal moves for the active player. Spectators receive an empty
     * array here.
     */
    validMoves: Move[];
    /**
     * Optional metadata about the transition from the previous state to
     * this one. This is intentionally lightweight and focused on UX-facing
     * summaries rather than full diffs.
     */
    meta?: {
      diffSummary?: {
        /**
         * Present when a pending PlayerChoice was auto-resolved by the
         * server as part of producing this update.
         */
        decisionAutoResolved?: DecisionAutoResolvedMeta;
      };
    };
  };
  /** ISO-8601 timestamp produced on the server. */
  timestamp: string;
}

/**
 * Canonical payload for `game_over` events.
 *
 * Emitted when a game reaches a terminal state (victory, draw,
 * abandonment, etc.).
 */
export interface GameOverMessage {
  type: 'game_over';
  data: {
    gameId: string;
    gameState: GameState;
    gameResult: GameResult;
  };
  timestamp: string;
}

/**
 * Per-player evaluation summary used for analysis/inspection views.
 *
 * totalEval is a zero-sum margin where positive values favour the player.
 * territoryEval and ringEval are optional decompositions of that margin.
 */
export interface PositionEvaluationByPlayer {
  totalEval: number;
  territoryEval?: number;
  ringEval?: number;
  winProbability?: number;
}

/**
 * Payload for streaming AI position evaluations over WebSocket.
 *
 * Emitted when analysis mode is enabled and a fresh GameState snapshot has
 * been evaluated by the strongest available engine.
 */
export interface PositionEvaluationPayload {
  type: 'position_evaluation';
  data: {
    gameId: string;
    moveNumber: number;
    boardType: BoardType;
    perPlayer: Record<number, PositionEvaluationByPlayer>;
    engineProfile: string;
    evaluationScale: 'zero_sum_margin' | 'win_probability';
  };
  timestamp: string;
}

/**
 * Payload for fatal AI / rules service failures that cause the game
 * to be abandoned.
 */
export interface GameErrorMessage {
  type: 'game_error';
  data: {
    message: string;
    /** Optional technical details intended for logs / debugging. */
    technical?: string;
    gameId: string;
  };
  timestamp: string;
}

/**
 * Lightweight diagnostic ping payload used for load testing and
 * latency measurement. This event is intentionally transport-only
 * and does not mutate game state.
 *
 * NOTE: The shape of this interface is kept in sync with
 * DiagnosticPingPayloadSchema in
 * src/shared/validation/websocketSchemas.ts so that runtime
 * validation and TypeScript contracts remain aligned.
 */
export interface DiagnosticPingPayload {
  /** Client-side timestamp in milliseconds since epoch. */
  timestamp: number;
  /**
   * Optional k6 virtual user identifier or similar diagnostic tag.
   * Accepts either a numeric VU id or an opaque string label.
   */
  vu?: string | number;
  /**
   * Optional monotonically increasing sequence number per connection,
   * used to correlate individual ping/pong pairs.
   */
  sequence?: number;
}

/**
 * Diagnostic pong payload echoed by the server. It carries the
 * original client payload plus a server-side timestamp so that
 * callers can compute round-trip latency.
 */
export interface DiagnosticPongPayload extends DiagnosticPingPayload {
  /** ISO-8601 timestamp produced on the server when the ping was handled. */
  serverTimestamp: string;
}

/**
 * Shared shape for per-player room notifications emitted on
 * `player_joined`, `player_left`, `player_disconnected`, and `player_reconnected`.
 */
export interface GamePlayerRoomEventPayload {
  type: 'player_joined' | 'player_left' | 'player_disconnected' | 'player_reconnected';
  data: {
    gameId: string;
    player: {
      id: string;
      username: string;
    };
  };
  timestamp: string;
}

export type PlayerJoinedPayload = GamePlayerRoomEventPayload & {
  type: 'player_joined';
};

export type PlayerLeftPayload = GamePlayerRoomEventPayload & {
  type: 'player_left';
};

export type PlayerDisconnectedPayload = GamePlayerRoomEventPayload & {
  type: 'player_disconnected';
};

export type PlayerReconnectedPayload = GamePlayerRoomEventPayload & {
  type: 'player_reconnected';
  data: GamePlayerRoomEventPayload['data'] & {
    playerNumber: number;
  };
};

/**
 * Payload for decision phase timeout warning events.
 *
 * Emitted to the player when they are approaching a decision phase timeout
 * (e.g., 5 seconds before auto-resolution).
 */
export interface DecisionPhaseTimeoutWarningPayload {
  type: 'decision_phase_timeout_warning';
  data: {
    gameId: string;
    playerNumber: number;
    phase: 'line_processing' | 'territory_processing' | 'chain_capture';
    remainingMs: number;
    choiceId?: string;
  };
  timestamp: string;
}

/**
 * Payload for decision phase timeout events.
 *
 * Emitted when a decision phase times out and is auto-resolved with a default
 * choice. Includes the chosen move so clients can update their state.
 */
export interface DecisionPhaseTimedOutPayload {
  type: 'decision_phase_timed_out';
  data: {
    gameId: string;
    playerNumber: number;
    phase: 'line_processing' | 'territory_processing' | 'chain_capture';
    /** The move ID that was auto-selected due to timeout */
    autoSelectedMoveId: string;
    /** Human-readable reason for the auto-selection */
    reason: string;
  };
  timestamp: string;
}

/**
 * Lobby broadcast payloads emitted via WebSocketServer.broadcastLobbyEvent
 * and consumed by LobbyPage.
 */
export type LobbyGameCreatedPayload = Game;

export interface LobbyGameJoinedPayload {
  gameId: string;
  playerCount: number;
}

export interface LobbyGameStartedPayload {
  gameId: string;
  status: Game['status'];
  startedAt: Game['startedAt'];
  playerCount: number;
}

export interface LobbyGameCancelledPayload {
  gameId: string;
}

/**
 * Payload for persisted chat messages with user info.
 * Used for both real-time messages and history retrieval.
 */
export interface ChatMessagePersisted {
  id: string;
  gameId: string;
  userId: string;
  username: string;
  message: string;
  createdAt: string; // ISO-8601
}

/**
 * Payload for chat history sent on game join.
 */
export interface ChatHistoryPayload {
  gameId: string;
  messages: ChatMessagePersisted[];
}

/**
 * Payload for rematch request events.
 */
export interface RematchRequestPayload {
  id: string;
  gameId: string;
  requesterId: string;
  requesterUsername: string;
  expiresAt: string; // ISO-8601
}

/**
 * Payload for rematch response events.
 */
export interface RematchResponsePayload {
  requestId: string;
  gameId: string;
  status: 'accepted' | 'declined' | 'expired';
  newGameId?: string;
}

/**
 * Client-to-server payload for requesting a rematch.
 */
export interface RematchRequestClientPayload {
  gameId: string;
}

/**
 * Client-to-server payload for responding to a rematch.
 */
export interface RematchResponseClientPayload {
  requestId: string;
  accept: boolean;
}

/**
 * Events the server can emit to connected clients over the game /
 * lobby sockets.
 */
export interface ServerToClientEvents {
  // Core game stream
  game_state: (payload: GameStateUpdateMessage) => void;
  game_over: (payload: GameOverMessage) => void;
  game_error: (payload: GameErrorMessage) => void;

  // Room-level player presence notifications
  player_joined: (payload: PlayerJoinedPayload) => void;
  player_left: (payload: PlayerLeftPayload) => void;
  player_disconnected: (payload: PlayerDisconnectedPayload) => void;
  player_reconnected: (payload: PlayerReconnectedPayload) => void;

  // Chat
  chat_message: (payload: ChatMessageServerPayload) => void;
  chat_message_persisted: (payload: ChatMessagePersisted) => void;
  chat_history: (payload: ChatHistoryPayload) => void;

  // Rematch system
  rematch_requested: (payload: RematchRequestPayload) => void;
  rematch_response: (payload: RematchResponsePayload) => void;

  // Legacy/experimental time control update event emitted from GameSession.
  // Currently not consumed by the React client but kept in the contract
  // to describe the runtime surface accurately.
  time_update: (payload: { playerId: string; playerNumber: number; timeRemaining: number }) => void;

  // Choice system
  player_choice_required: (choice: PlayerChoice) => void;
  player_choice_canceled: (choiceId: string) => void;

  // Decision phase timeout events
  decision_phase_timeout_warning: (payload: DecisionPhaseTimeoutWarningPayload) => void;
  decision_phase_timed_out: (payload: DecisionPhaseTimedOutPayload) => void;

  // Structured transport-level errors
  error: (payload: WebSocketErrorPayload) => void;

  // Optional analysis-mode evaluation stream; when enabled, the server emits
  // best-effort position_evaluation events after moves for inspection UIs.
  position_evaluation?: (payload: PositionEvaluationPayload) => void;

  // Diagnostic: load-testing ping/pong channel. This is transport-only and
  // does not mutate game state.
  'diagnostic:pong': (payload: DiagnosticPongPayload) => void;

  // Lobby broadcasts
  'lobby:game_created': (payload: LobbyGameCreatedPayload) => void;
  'lobby:game_joined': (payload: LobbyGameJoinedPayload) => void;
  'lobby:game_started': (payload: LobbyGameStartedPayload) => void;
  'lobby:game_cancelled': (payload: LobbyGameCancelledPayload) => void;

  // Matchmaking events
  'match-found': (payload: MatchFoundPayload) => void;
  'matchmaking-status': (payload: MatchmakingStatus) => void;

  // Reserved for future use: explicit reconnect / resync request.
  // Currently listened to by GameContext but not emitted by the server.
  request_reconnect?: () => void;
}

/**
 * Events that clients are allowed to emit to the server.
 *
 * Payload types for the core game events are derived from the Zod
 * schemas in src/shared/validation/websocketSchemas.ts to keep the
 * runtime validators and TypeScript contracts aligned.
 */
export interface ClientToServerEvents {
  // Game lifecycle / room membership
  join_game: (payload: JoinGamePayload) => void;
  leave_game: (payload: LeaveGamePayload) => void;

  // Move submission
  player_move: (payload: PlayerMovePayload) => void;
  player_move_by_id: (payload: PlayerMoveByIdPayload) => void;

  // Choice system
  player_choice_response: (payload: PlayerChoiceResponsePayload) => void;

  // Chat
  chat_message: (payload: ChatMessagePayload) => void;

  // Rematch system
  rematch_request: (payload: RematchRequestClientPayload) => void;
  rematch_respond: (payload: RematchResponseClientPayload) => void;

  // Diagnostic: load-testing ping channel
  'diagnostic:ping': (payload: DiagnosticPingPayload) => void;

  // Lobby subscription
  'lobby:subscribe': () => void;
  'lobby:unsubscribe': () => void;
}

/**
 * Convenience aliases for event names, useful when constraining
 * helper utilities or logging.
 */
export type ServerToClientEventName = keyof ServerToClientEvents;
export type ClientToServerEventName = keyof ClientToServerEvents;

// Re-export payload types from the Zod schema module so that callers
// can treat this file as the single source of truth for WebSocket
// contracts without importing validation code directly.
export type {
  JoinGamePayload,
  LeaveGamePayload,
  PlayerMovePayload,
  PlayerMoveByIdPayload,
  ChatMessagePayload,
  PlayerChoiceResponsePayload,
} from '../validation/websocketSchemas';
