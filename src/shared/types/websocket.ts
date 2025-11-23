import type { Game, GameResult, GameState, Move, PlayerChoice } from './game';
import type {
  JoinGamePayload,
  LeaveGamePayload,
  PlayerMovePayload,
  PlayerMoveByIdPayload,
  ChatMessagePayload,
  PlayerChoiceResponsePayload,
} from '../validation/websocketSchemas';

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
  | 'MOVE_REJECTED'
  | 'CHOICE_REJECTED'
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
 * Shared shape for per-player room notifications emitted on
 * `player_joined`, `player_left`, and `player_disconnected`.
 */
export interface GamePlayerRoomEventPayload {
  type: 'player_joined' | 'player_left' | 'player_disconnected';
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

  // Chat
  chat_message: (payload: ChatMessageServerPayload) => void;

  // Legacy/experimental time control update event emitted from GameSession.
  // Currently not consumed by the React client but kept in the contract
  // to describe the runtime surface accurately.
  time_update: (payload: { playerId: string; playerNumber: number; timeRemaining: number }) => void;

  // Choice system
  player_choice_required: (choice: PlayerChoice) => void;
  player_choice_canceled: (choiceId: string) => void;

  // Structured transport-level errors
  error: (payload: WebSocketErrorPayload) => void;

  // Lobby broadcasts
  'lobby:game_created': (payload: LobbyGameCreatedPayload) => void;
  'lobby:game_joined': (payload: LobbyGameJoinedPayload) => void;
  'lobby:game_started': (payload: LobbyGameStartedPayload) => void;
  'lobby:game_cancelled': (payload: LobbyGameCancelledPayload) => void;

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
