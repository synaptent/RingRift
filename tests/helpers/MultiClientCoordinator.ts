/**
 * MultiClientCoordinator - Multi-context WebSocket Coordination Helper
 * ============================================================================
 *
 * A test utility class for managing multiple WebSocket client connections
 * in E2E tests. Enables coordinated actions between multiple players in
 * multiplayer game scenarios.
 *
 * Features:
 * - Manage multiple Socket.IO client connections
 * - Coordinate actions between clients (e.g., player 1 moves, player 2 responds)
 * - Promise-based waiting for specific game events
 * - Message queuing per client for inspection
 * - Robust cleanup of all connections after tests
 *
 * @example
 * ```typescript
 * const coordinator = new MultiClientCoordinator('http://localhost:3000');
 *
 * // Connect two players
 * await coordinator.connect('player1', { playerId: 'p1', token: 'jwt-token-1' });
 * await coordinator.connect('player2', { playerId: 'p2', token: 'jwt-token-2' });
 *
 * // Join a game
 * await coordinator.send('player1', { type: 'join_game', gameId: 'game-123' });
 * await coordinator.send('player2', { type: 'join_game', gameId: 'game-123' });
 *
 * // Wait for both to receive game state
 * await coordinator.waitForAll(['player1', 'player2'], {
 *   type: 'event',
 *   eventName: 'game_state',
 *   predicate: (data) => data.data?.gameId === 'game-123'
 * });
 *
 * // Cleanup
 * await coordinator.cleanup();
 * ```
 */

import { io, Socket } from 'socket.io-client';
import type {
  ServerToClientEvents,
  ClientToServerEvents,
  GameStateUpdateMessage,
  GameOverMessage,
  WebSocketErrorPayload,
} from '../../src/shared/types/websocket';
import type { GamePhase, GameState } from '../../src/shared/types/game';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for connecting a client.
 */
export interface ClientConfig {
  /** Player ID for this client */
  playerId: string;
  /** Optional user ID (for authenticated sessions) */
  userId?: string;
  /** JWT authentication token */
  token: string;
  /** Optional extra Socket.IO options */
  socketOptions?: Partial<Parameters<typeof io>[1]>;
}

/**
 * Condition for waiting on a specific event or state.
 */
export interface WaitCondition {
  /** Type of condition to wait for */
  type: 'gameState' | 'phase' | 'turn' | 'event' | 'gameOver';
  /** Predicate function to match the condition */
  predicate: (data: unknown) => boolean;
  /** Timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Optional specific event name to listen for */
  eventName?: keyof ServerToClientEvents;
}

/**
 * Captured message with metadata.
 */
export interface CapturedMessage {
  eventName: string;
  payload: unknown;
  timestamp: number;
}

/**
 * Action step for sequential execution.
 */
export interface ActionStep {
  /** Client ID to perform the action on */
  clientId: string;
  /** Action to perform (async) */
  action: () => Promise<void>;
  /** Optional condition to wait for after action */
  waitFor?: WaitCondition;
  /** Optional client ID to check condition on (defaults to actionclientId) */
  waitOnClientId?: string;
}

/**
 * Per-client connection state.
 */
interface ClientState {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  config: ClientConfig;
  messageQueue: CapturedMessage[];
  pendingWaits: Map<
    string,
    {
      condition: WaitCondition;
      resolve: (data: unknown) => void;
      reject: (error: Error) => void;
      timeoutId: NodeJS.Timeout;
    }
  >;
}

// ============================================================================
// MultiClientCoordinator Class
// ============================================================================

/**
 * Coordinates multiple WebSocket client connections for E2E testing.
 */
export class MultiClientCoordinator {
  private clients: Map<string, ClientState> = new Map();
  private serverUrl: string;
  private defaultTimeout: number;

  /**
   * Creates a new MultiClientCoordinator.
   *
   * @param serverUrl - The WebSocket server URL (e.g., 'http://localhost:3000')
   * @param defaultTimeout - Default timeout for waits in milliseconds (default: 30000)
   */
  constructor(serverUrl: string, defaultTimeout = 30000) {
    this.serverUrl = serverUrl;
    this.defaultTimeout = defaultTimeout;
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Connects a new client to the WebSocket server.
   *
   * @param clientId - Unique identifier for this client in the coordinator
   * @param config - Client configuration including auth token
   * @returns Promise that resolves when connected
   */
  async connect(clientId: string, config: ClientConfig): Promise<void> {
    if (this.clients.has(clientId)) {
      throw new Error(`Client '${clientId}' is already connected`);
    }

    const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io(this.serverUrl, {
      transports: ['websocket', 'polling'],
      auth: { token: config.token },
      autoConnect: false,
      ...config.socketOptions,
    });

    const clientState: ClientState = {
      socket,
      config,
      messageQueue: [],
      pendingWaits: new Map(),
    };

    this.clients.set(clientId, clientState);
    this.setupMessageHandler(clientId, clientState);

    // Wait for connection
    await new Promise<void>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Connection timeout for client '${clientId}'`));
      }, this.defaultTimeout);

      socket.on('connect', () => {
        clearTimeout(timeoutId);
        resolve();
      });

      socket.on('connect_error', (err) => {
        clearTimeout(timeoutId);
        reject(new Error(`Connection error for client '${clientId}': ${err.message}`));
      });

      socket.connect();
    });
  }

  /**
   * Disconnects a specific client.
   *
   * @param clientId - ID of the client to disconnect
   */
  async disconnect(clientId: string): Promise<void> {
    const state = this.clients.get(clientId);
    if (!state) {
      return;
    }

    // Cancel all pending waits
    for (const [, pending] of state.pendingWaits) {
      clearTimeout(pending.timeoutId);
      pending.reject(new Error(`Client '${clientId}' disconnected`));
    }
    state.pendingWaits.clear();

    // Disconnect socket
    state.socket.disconnect();
    this.clients.delete(clientId);
  }

  /** Reconnects a force-disconnected client while preserving its handlers. */
  async reconnect(clientId: string): Promise<void> {
    const state = this.clients.get(clientId);
    if (!state) {
      throw new Error(`Client '${clientId}' is not registered`);
    }
    if (state.socket.connected) {
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        state.socket.off('connect', handleConnect);
        state.socket.off('connect_error', handleError);
        reject(new Error(`Reconnection timeout for client '${clientId}'`));
      }, this.defaultTimeout);
      const handleConnect = () => {
        clearTimeout(timeoutId);
        state.socket.off('connect_error', handleError);
        resolve();
      };
      const handleError = (err: Error) => {
        clearTimeout(timeoutId);
        state.socket.off('connect', handleConnect);
        reject(new Error(`Reconnection error for client '${clientId}': ${err.message}`));
      };

      state.socket.once('connect', handleConnect);
      state.socket.once('connect_error', handleError);
      state.socket.connect();
    });
  }

  /**
   * Disconnects all connected clients.
   */
  async disconnectAll(): Promise<void> {
    const errors: Error[] = [];
    for (const clientId of this.clients.keys()) {
      try {
        await this.disconnect(clientId);
      } catch (e) {
        errors.push(e instanceof Error ? e : new Error(String(e)));
      }
    }
    if (errors.length > 0) {
      console.warn('Errors during disconnectAll:', errors);
    }
  }

  /**
   * Cleans up all connections and resets state.
   * Alias for disconnectAll with error suppression for use in afterEach.
   */
  async cleanup(): Promise<void> {
    const errors: Error[] = [];
    for (const [clientId, state] of this.clients) {
      try {
        // Clear pending waits
        for (const [, pending] of state.pendingWaits) {
          clearTimeout(pending.timeoutId);
        }
        state.pendingWaits.clear();
        state.socket.disconnect();
      } catch (e) {
        errors.push(
          e instanceof Error ? e : new Error(`Cleanup error for ${clientId}: ${String(e)}`)
        );
      }
    }
    this.clients.clear();
    if (errors.length > 0) {
      console.warn('Cleanup errors:', errors);
    }
  }

  // ==========================================================================
  // Message Handling
  // ==========================================================================

  /**
   * Sets up message handlers for a client to capture events.
   */
  private setupMessageHandler(clientId: string, state: ClientState): void {
    const { socket } = state;

    // List of server events to capture
    const eventsToCapture: (keyof ServerToClientEvents)[] = [
      'game_state',
      'game_over',
      'game_error',
      'player_joined',
      'player_left',
      'player_disconnected',
      'player_reconnected',
      'chat_message',
      'chat_message_persisted',
      'chat_history',
      'player_choice_required',
      'player_choice_canceled',
      'decision_phase_timeout_warning',
      'decision_phase_timed_out',
      'error',
      'rematch_requested',
      'rematch_response',
      'time_update',
    ];

    for (const eventName of eventsToCapture) {
      socket.on(eventName, (payload: unknown) => {
        const message: CapturedMessage = {
          eventName,
          payload,
          timestamp: Date.now(),
        };
        state.messageQueue.push(message);
        this.checkPendingWaits(clientId, state, eventName, payload);
      });
    }
  }

  /**
   * Checks if any pending waits are satisfied by a new message.
   */
  private checkPendingWaits(
    clientId: string,
    state: ClientState,
    eventName: string,
    payload: unknown
  ): void {
    for (const [waitId, pending] of state.pendingWaits) {
      const { condition, resolve, timeoutId } = pending;
      try {
        if (this.conditionMatches(condition, eventName, payload)) {
          clearTimeout(timeoutId);
          state.pendingWaits.delete(waitId);
          resolve(payload);
        }
      } catch (e) {
        // Predicate threw, ignore
      }
    }
  }

  /**
   * Checks if a condition matches an event.
   */
  private conditionMatches(condition: WaitCondition, eventName: string, payload: unknown): boolean {
    switch (condition.type) {
      case 'event':
        if (condition.eventName && eventName !== condition.eventName) {
          return false;
        }
        return condition.predicate(payload);

      case 'gameState':
        if (eventName !== 'game_state') return false;
        return condition.predicate(payload);

      case 'gameOver':
        if (eventName !== 'game_over') return false;
        return condition.predicate(payload);

      case 'phase':
        if (eventName !== 'game_state') return false;
        return condition.predicate(payload);

      case 'turn':
        if (eventName !== 'game_state') return false;
        return condition.predicate(payload);

      default:
        return false;
    }
  }

  // ==========================================================================
  // Sending Messages
  // ==========================================================================

  /**
   * Sends a message from a specific client.
   *
   * @param clientId - ID of the client to send from
   * @param event - Event name
   * @param payload - Event payload
   */
  async send<E extends keyof ClientToServerEvents>(
    clientId: string,
    event: E,
    payload: Parameters<ClientToServerEvents[E]>[0]
  ): Promise<void> {
    const state = this.clients.get(clientId);
    if (!state) {
      throw new Error(`Client '${clientId}' is not connected`);
    }

    // Type assertion needed due to Socket.IO's complex typing
    (state.socket as any).emit(event, payload);
  }

  /**
   * Sends a join_game event.
   */
  async joinGame(clientId: string, gameId: string): Promise<void> {
    await this.send(clientId, 'join_game', { gameId });
  }

  /**
   * Sends a leave_game event.
   */
  async leaveGame(clientId: string, gameId: string): Promise<void> {
    await this.send(clientId, 'leave_game', { gameId });
  }

  /**
   * Sends a player_move_by_id event.
   */
  async sendMoveById(clientId: string, gameId: string, moveId: string): Promise<void> {
    await this.send(clientId, 'player_move_by_id', { gameId, moveId });
  }

  /**
   * Sends a chat message.
   */
  async sendChat(clientId: string, gameId: string, text: string): Promise<void> {
    await this.send(clientId, 'chat_message', { gameId, text });
  }

  // ==========================================================================
  // Waiting for Events
  // ==========================================================================

  /**
   * Waits for a condition to be met on a specific client.
   *
   * @param clientId - ID of the client to wait on
   * @param condition - Condition to wait for
   * @returns Promise that resolves with the matching data
   */
  async waitFor(clientId: string, condition: WaitCondition): Promise<unknown> {
    const state = this.clients.get(clientId);
    if (!state) {
      throw new Error(`Client '${clientId}' is not connected`);
    }

    const timeout = condition.timeout ?? this.defaultTimeout;
    const waitId = `${Date.now()}-${Math.random()}`;

    // First check existing messages in queue
    for (const message of state.messageQueue) {
      if (this.conditionMatches(condition, message.eventName, message.payload)) {
        return message.payload;
      }
    }

    // Set up wait for future messages
    return new Promise<unknown>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        state.pendingWaits.delete(waitId);
        reject(
          new Error(
            `Timeout waiting for condition on client '${clientId}' (type: ${condition.type}, timeout: ${timeout}ms)`
          )
        );
      }, timeout);

      state.pendingWaits.set(waitId, { condition, resolve, reject, timeoutId });
    });
  }

  /**
   * Waits for the same condition to be met on multiple clients.
   *
   * @param clientIds - IDs of clients to wait on
   * @param condition - Condition to wait for
   * @returns Promise that resolves with a map of clientId -> matching data
   */
  async waitForAll(clientIds: string[], condition: WaitCondition): Promise<Map<string, unknown>> {
    const results = new Map<string, unknown>();
    await Promise.all(
      clientIds.map(async (clientId) => {
        const data = await this.waitFor(clientId, condition);
        results.set(clientId, data);
      })
    );
    return results;
  }

  /**
   * Waits for different conditions on different clients.
   *
   * @param conditions - Map of clientId -> condition
   * @returns Promise that resolves with a map of clientId -> matching data
   */
  async waitForMany(conditions: Map<string, WaitCondition>): Promise<Map<string, unknown>> {
    const results = new Map<string, unknown>();
    await Promise.all(
      Array.from(conditions.entries()).map(async ([clientId, condition]) => {
        const data = await this.waitFor(clientId, condition);
        results.set(clientId, data);
      })
    );
    return results;
  }

  // ==========================================================================
  // Convenience Wait Methods
  // ==========================================================================

  /**
   * Waits for a game state update matching a predicate.
   */
  async waitForGameState(
    clientId: string,
    predicate: (state: GameState) => boolean,
    timeout?: number
  ): Promise<GameStateUpdateMessage> {
    return (await this.waitFor(clientId, {
      type: 'gameState',
      predicate: (data) => {
        const msg = data as GameStateUpdateMessage;
        return msg?.data?.gameState ? predicate(msg.data.gameState) : false;
      },
      timeout,
    })) as GameStateUpdateMessage;
  }

  /**
   * Waits for the game to reach a specific phase.
   */
  async waitForPhase(
    clientId: string,
    phase: GamePhase,
    timeout?: number
  ): Promise<GameStateUpdateMessage> {
    return this.waitForGameState(clientId, (state) => state.currentPhase === phase, timeout);
  }

  /**
   * Waits for a specific player's turn.
   */
  async waitForTurn(
    clientId: string,
    playerNumber: number,
    timeout?: number
  ): Promise<GameStateUpdateMessage> {
    return this.waitForGameState(
      clientId,
      (state) => state.currentPlayer === playerNumber,
      timeout
    );
  }

  /**
   * Waits for the game to end.
   */
  async waitForGameOver(clientId: string, timeout?: number): Promise<GameOverMessage> {
    return (await this.waitFor(clientId, {
      type: 'gameOver',
      predicate: () => true,
      timeout,
    })) as GameOverMessage;
  }

  /**
   * Waits for a specific event by name.
   */
  async waitForEvent(
    clientId: string,
    eventName: keyof ServerToClientEvents,
    predicate?: (payload: unknown) => boolean,
    timeout?: number
  ): Promise<unknown> {
    return await this.waitFor(clientId, {
      type: 'event',
      eventName,
      predicate: predicate ?? (() => true),
      timeout,
    });
  }

  // ==========================================================================
  // Coordination Primitives
  // ==========================================================================

  /**
   * Executes a sequence of actions, optionally waiting for conditions between steps.
   *
   * @param actions - Array of action steps to execute in order
   */
  async executeSequence(actions: ActionStep[]): Promise<void> {
    for (const step of actions) {
      await step.action();
      if (step.waitFor) {
        const waitOnClient = step.waitOnClientId ?? step.clientId;
        await this.waitFor(waitOnClient, step.waitFor);
      }
    }
  }

  /**
   * Executes actions in parallel and waits for all to complete.
   *
   * @param actions - Array of async actions to execute
   */
  async executeParallel(actions: Array<() => Promise<void>>): Promise<void> {
    await Promise.all(actions.map((action) => action()));
  }

  // ==========================================================================
  // Message Queue Access
  // ==========================================================================

  /**
   * Gets all captured messages for a client.
   *
   * @param clientId - ID of the client
   * @returns Array of captured messages
   */
  getMessages(clientId: string): CapturedMessage[] {
    const state = this.clients.get(clientId);
    if (!state) {
      throw new Error(`Client '${clientId}' is not connected`);
    }
    return [...state.messageQueue];
  }

  /**
   * Clears the message queue for a client.
   *
   * @param clientId - ID of the client
   */
  clearMessages(clientId: string): void {
    const state = this.clients.get(clientId);
    if (!state) {
      throw new Error(`Client '${clientId}' is not connected`);
    }
    state.messageQueue = [];
  }

  /**
   * Gets messages matching a filter.
   *
   * @param clientId - ID of the client
   * @param filter - Filter function
   * @returns Array of matching messages
   */
  getMessagesMatching(
    clientId: string,
    filter: (msg: CapturedMessage) => boolean
  ): CapturedMessage[] {
    return this.getMessages(clientId).filter(filter);
  }

  /**
   * Gets the last received game state for a client.
   *
   * @param clientId - ID of the client
   * @returns The most recent game state, or null if none received
   */
  getLastGameState(clientId: string): GameState | null {
    const messages = this.getMessagesMatching(clientId, (msg) => msg.eventName === 'game_state');
    if (messages.length === 0) return null;
    const lastMessage = messages[messages.length - 1];
    const payload = lastMessage.payload as GameStateUpdateMessage;
    return payload?.data?.gameState ?? null;
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Gets the list of connected client IDs.
   */
  getConnectedClientIds(): string[] {
    return Array.from(this.clients.keys());
  }

  /**
   * Checks if a client is connected.
   */
  isConnected(clientId: string): boolean {
    const state = this.clients.get(clientId);
    return state?.socket.connected ?? false;
  }

  /**
   * Gets the underlying socket for a client (for advanced use cases).
   */
  getSocket(clientId: string): Socket<ServerToClientEvents, ClientToServerEvents> | null {
    return this.clients.get(clientId)?.socket ?? null;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Creates a new MultiClientCoordinator with default settings.
 *
 * @param serverUrl - The WebSocket server URL
 * @returns A new MultiClientCoordinator instance
 */
export function createMultiClientCoordinator(serverUrl: string): MultiClientCoordinator {
  return new MultiClientCoordinator(serverUrl);
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guard for GameStateUpdateMessage.
 */
export function isGameStateMessage(msg: unknown): msg is GameStateUpdateMessage {
  return (
    typeof msg === 'object' &&
    msg !== null &&
    (msg as any).type === 'game_update' &&
    (msg as any).data?.gameState !== undefined
  );
}

/**
 * Type guard for GameOverMessage.
 */
export function isGameOverMessage(msg: unknown): msg is GameOverMessage {
  return (
    typeof msg === 'object' &&
    msg !== null &&
    (msg as any).type === 'game_over' &&
    (msg as any).data?.gameResult !== undefined
  );
}

/**
 * Type guard for WebSocketErrorPayload.
 */
export function isErrorPayload(msg: unknown): msg is WebSocketErrorPayload {
  return (
    typeof msg === 'object' &&
    msg !== null &&
    (msg as any).type === 'error' &&
    typeof (msg as any).code === 'string'
  );
}
