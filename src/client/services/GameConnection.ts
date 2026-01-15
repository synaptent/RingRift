import { io, Socket } from 'socket.io-client';
import { getSocketBaseUrl } from '../utils/socketBaseUrl';
import type {
  GameStateUpdateMessage,
  GameOverMessage,
  ChatMessageServerPayload,
  ChatMessagePersisted,
  ChatHistoryPayload,
  JoinGamePayload,
  PlayerMovePayload,
  PlayerChoiceResponsePayload,
  ChatMessagePayload,
  WebSocketErrorPayload,
  DecisionPhaseTimeoutWarningPayload,
  DecisionPhaseTimedOutPayload,
  RematchRequestPayload,
  RematchResponsePayload,
  RematchRequestClientPayload,
  RematchResponseClientPayload,
  PositionEvaluationPayload,
  PlayerDisconnectedPayload,
  PlayerReconnectedPayload,
} from '../../shared/types/websocket';
import type { Move, PlayerChoice } from '../../shared/types/game';
import type { GameConnection, GameEventHandlers, ConnectionStatus } from '../domain/GameAPI';

/**
 * Socket.IO-backed implementation of the GameConnection domain API.
 *
 * This class encapsulates all direct Socket.IO usage so that
 * React contexts and components can depend on the small
 * `GameConnection` interface instead of raw sockets.
 */
export class SocketGameConnection implements GameConnection {
  private socket: Socket | null = null;
  private _gameId: string | null = null;
  private _status: ConnectionStatus = 'disconnected';
  private readonly handlers: GameEventHandlers;

  constructor(handlers: GameEventHandlers) {
    this.handlers = handlers;
  }

  get gameId(): string | null {
    return this._gameId;
  }

  get status(): ConnectionStatus {
    return this._status;
  }

  private setStatus(status: ConnectionStatus): void {
    this._status = status;
    if (this.handlers.onConnectionStatusChange) {
      this.handlers.onConnectionStatusChange(status);
    }
  }

  async connect(targetGameId: string): Promise<void> {
    // If already connected to this game, do nothing.
    if (this._gameId === targetGameId && this.socket) {
      return;
    }

    this.disconnect();
    this.setStatus('connecting');

    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    const baseUrl = getSocketBaseUrl();

    const socketOptions: Parameters<typeof io>[1] = {
      transports: ['websocket', 'polling'],
    };
    if (token) {
      socketOptions.auth = { token };
    }

    const socket = io(baseUrl, socketOptions);

    this.socket = socket;

    const emitJoinGame = () => {
      const payload: JoinGamePayload = { gameId: targetGameId };
      socket.emit('join_game', payload);
    };

    socket.on('connect', () => {
      this._gameId = targetGameId;
      this.setStatus('connected');
      emitJoinGame();
    });

    socket.on('connect_error', (err: Error) => {
      this.handlers.onError(err);
      this.setStatus('disconnected');
    });

    socket.on('reconnect_attempt', () => {
      this.setStatus('reconnecting');
    });

    socket.on('reconnect', () => {
      this.setStatus('connected');
      emitJoinGame();
    });

    socket.on('game_state', (payload: GameStateUpdateMessage) => {
      this.handlers.onGameState(payload);
      this.setStatus('connected');
    });

    socket.on('game_over', (payload: GameOverMessage) => {
      this.handlers.onGameOver(payload);
    });

    socket.on('player_choice_required', (choice: PlayerChoice) => {
      this.handlers.onChoiceRequired(choice);
    });

    socket.on('player_choice_canceled', (choiceId: string) => {
      this.handlers.onChoiceCanceled(choiceId);
    });

    socket.on('chat_message', (payload: ChatMessageServerPayload) => {
      this.handlers.onChatMessage(payload);
    });

    socket.on('chat_message_persisted', (payload: ChatMessagePersisted) => {
      if (this.handlers.onChatMessagePersisted) {
        this.handlers.onChatMessagePersisted(payload);
      }
    });

    socket.on('chat_history', (payload: ChatHistoryPayload) => {
      if (this.handlers.onChatHistory) {
        this.handlers.onChatHistory(payload);
      }
    });

    socket.on('decision_phase_timeout_warning', (payload: DecisionPhaseTimeoutWarningPayload) => {
      if (this.handlers.onDecisionPhaseTimeoutWarning) {
        this.handlers.onDecisionPhaseTimeoutWarning(payload);
      }
    });

    socket.on('decision_phase_timed_out', (payload: DecisionPhaseTimedOutPayload) => {
      if (this.handlers.onDecisionPhaseTimedOut) {
        this.handlers.onDecisionPhaseTimedOut(payload);
      }
    });

    socket.on('rematch_requested', (payload: RematchRequestPayload) => {
      if (this.handlers.onRematchRequested) {
        this.handlers.onRematchRequested(payload);
      }
    });

    socket.on('rematch_response', (payload: RematchResponsePayload) => {
      if (this.handlers.onRematchResponse) {
        this.handlers.onRematchResponse(payload);
      }
    });

    socket.on('position_evaluation', (payload: PositionEvaluationPayload) => {
      if (this.handlers.onPositionEvaluation) {
        this.handlers.onPositionEvaluation(payload);
      }
    });

    socket.on('player_disconnected', (payload: PlayerDisconnectedPayload) => {
      if (this.handlers.onPlayerDisconnected) {
        this.handlers.onPlayerDisconnected(payload);
      }
    });

    socket.on('player_reconnected', (payload: PlayerReconnectedPayload) => {
      if (this.handlers.onPlayerReconnected) {
        this.handlers.onPlayerReconnected(payload);
      }
    });

    socket.on('error', (payload: WebSocketErrorPayload) => {
      this.handlers.onError(payload);
    });

    socket.on('disconnect', (reason) => {
      this.handlers.onDisconnect(reason);
      this.setStatus('disconnected');
      // Keep socket instance to allow Socket.IO auto-reconnect unless
      // we explicitly called disconnect().
      if (reason === 'io client disconnect') {
        this.socket = null;
      }
    });

    // Optional explicit reconnect / resync request from the server.
    socket.on('request_reconnect', () => {
      emitJoinGame();
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this._gameId = null;
    this.setStatus('disconnected');
  }

  submitMove(move: Move): void {
    if (!this.socket || !this._gameId) {
      console.warn('submitMove called without active socket/game');
      return;
    }

    // Extract captureTarget for capture moves (may be on move object)
    const captureTarget =
      'captureTarget' in move ? (move as { captureTarget?: unknown }).captureTarget : undefined;

    const payload: PlayerMovePayload = {
      gameId: this._gameId,
      move: {
        moveNumber: move.moveNumber ?? 0,
        position: JSON.stringify({
          from: move.from,
          to: move.to,
          // Include placement-specific fields for ring placement moves
          placementCount: move.placementCount,
          placedOnStack: move.placedOnStack,
          // Include capture-specific fields for capture moves
          captureTarget,
        }),
        moveType: move.type as PlayerMovePayload['move']['moveType'],
      },
    };

    this.socket.emit('player_move', payload);
  }

  respondToChoice(choice: PlayerChoice, selectedOption: unknown): void {
    if (!this.socket || !this._gameId) {
      console.warn('respondToChoice called without active socket/game');
      return;
    }

    // RR-FIX-2026-01-15: Always use player_choice_response instead of player_move_by_id.
    // The player_move_by_id path causes lock contention when the server is awaiting
    // a choice response (the game lock is held during requestChoice, so player_move_by_id
    // cannot acquire it). player_choice_response doesn't need the lock - it just
    // resolves the pending Promise.
    const response: PlayerChoiceResponsePayload = {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      choiceType: choice.type as PlayerChoiceResponsePayload['choiceType'],
      selectedOption,
    };

    this.socket.emit('player_choice_response', response);
  }

  sendChatMessage(text: string): void {
    if (!this.socket || !this._gameId) {
      console.warn('sendChatMessage called without active socket/game');
      return;
    }

    const payload = {
      gameId: this._gameId,
      text,
    } as ChatMessagePayload;

    this.socket.emit('chat_message', payload);
  }

  requestRematch(): void {
    if (!this.socket || !this._gameId) {
      console.warn('requestRematch called without active socket/game');
      return;
    }

    const payload: RematchRequestClientPayload = {
      gameId: this._gameId,
    };

    this.socket.emit('rematch_request', payload);
  }

  respondToRematch(requestId: string, accept: boolean): void {
    if (!this.socket) {
      console.warn('respondToRematch called without active socket');
      return;
    }

    const payload: RematchResponseClientPayload = {
      requestId,
      accept,
    };

    this.socket.emit('rematch_respond', payload);
  }
}
