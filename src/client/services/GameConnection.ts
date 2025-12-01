import { io, Socket } from 'socket.io-client';
import { getSocketBaseUrl } from '../utils/socketBaseUrl';
import type {
  GameStateUpdateMessage,
  GameOverMessage,
  ChatMessageServerPayload,
  JoinGamePayload,
  PlayerMovePayload,
  PlayerMoveByIdPayload,
  PlayerChoiceResponsePayload,
  ChatMessagePayload,
  WebSocketErrorPayload,
  DecisionPhaseTimeoutWarningPayload,
  DecisionPhaseTimedOutPayload,
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

    const socket = io(baseUrl, {
      transports: ['websocket', 'polling'],
      auth: token ? { token } : undefined,
    });

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

    socket.on('error', (payload: WebSocketErrorPayload | any) => {
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
      // eslint-disable-next-line no-console
      console.warn('submitMove called without active socket/game');
      return;
    }

    const payload: PlayerMovePayload = {
      gameId: this._gameId,
      move: {
        moveNumber: move.moveNumber ?? 0,
        position: JSON.stringify({ from: move.from, to: move.to }),
        moveType: move.type as PlayerMovePayload['move']['moveType'],
      },
    };

    this.socket.emit('player_move', payload);
  }

  respondToChoice(choice: PlayerChoice, selectedOption: unknown): void {
    if (!this.socket || !this._gameId) {
      // eslint-disable-next-line no-console
      console.warn('respondToChoice called without active socket/game');
      return;
    }

    let moveId: string | undefined;

    if (
      choice.type === 'line_order' ||
      choice.type === 'region_order' ||
      choice.type === 'ring_elimination'
    ) {
      moveId =
        selectedOption && typeof (selectedOption as any).moveId === 'string'
          ? (selectedOption as any).moveId
          : undefined;
    } else if (choice.type === 'line_reward_option') {
      const optionKey = selectedOption as string;
      moveId = choice.moveIds?.[optionKey as keyof typeof choice.moveIds];
    }

    if (moveId) {
      const payload: PlayerMoveByIdPayload = { gameId: this._gameId, moveId };
      this.socket.emit('player_move_by_id', payload);
      return;
    }

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
      // eslint-disable-next-line no-console
      console.warn('sendChatMessage called without active socket/game');
      return;
    }

    const payload = {
      gameId: this._gameId,
      text,
    } as ChatMessagePayload;

    this.socket.emit('chat_message', payload);
  }
}
