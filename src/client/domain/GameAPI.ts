import type { GameState, Move, PlayerChoice, GameResult } from '../../shared/types/game';
import type {
  ChatMessageServerPayload,
  GameStateUpdateMessage,
  GameOverMessage,
  WebSocketErrorPayload,
  DecisionPhaseTimeoutWarningPayload,
  DecisionPhaseTimedOutPayload,
} from '../../shared/types/websocket';

/**
 * Thin, host-agnostic domain API surface for client-side game logic.
 *
 * This module defines the minimal callback/command interface that
 * client code uses to interact with an underlying game session,
 * without coupling components directly to Socket.IO or React state.
 *
 * Initial scope:
 * - WebSocket-style event callbacks (state, game over, error, chat).
 * - Commands to connect/disconnect, submit moves, respond to choices,
 *   and send chat messages.
 *
 * Future scope:
 * - Support for offline/sandbox engines behind the same interface.
 * - Pluggable transports (WebSocket, WebRTC, local mocks).
 */

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

export interface GameEventHandlers {
  onGameState: (message: GameStateUpdateMessage) => void;
  onGameOver: (message: GameOverMessage) => void;
  onChoiceRequired: (choice: PlayerChoice) => void;
  onChoiceCanceled: (choiceId: string) => void;
  onChatMessage: (payload: ChatMessageServerPayload) => void;
  onError: (payload: WebSocketErrorPayload | unknown) => void;
  onDisconnect: (reason: string) => void;
  onConnectionStatusChange?: (status: ConnectionStatus) => void;
  /** Optional handler for decision-phase timeout warnings (countdown UX). */
  onDecisionPhaseTimeoutWarning?: (payload: DecisionPhaseTimeoutWarningPayload) => void;
  /** Optional handler for final decision-phase timeout + auto-resolve events. */
  onDecisionPhaseTimedOut?: (payload: DecisionPhaseTimedOutPayload) => void;
}

export interface GameConnection {
  /** Identifier for the connected game, if any. */
  readonly gameId: string | null;
  /** Current connection status. */
  readonly status: ConnectionStatus;

  /** Establish a connection to the given game id. */
  connect(gameId: string): Promise<void>;

  /** Disconnect from the current game and release resources. */
  disconnect(): void;

  /** Submit a canonical Move to the backend. */
  submitMove(move: Move): void;

  /** Respond to a pending PlayerChoice. */
  respondToChoice(choice: PlayerChoice, selectedOption: unknown): void;

  /** Send a chat message for the current game. */
  sendChatMessage(text: string): void;
}
