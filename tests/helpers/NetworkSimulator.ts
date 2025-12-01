/**
 * NetworkSimulator - Network Partition Simulation for E2E Reconnection Tests
 * ============================================================================
 *
 * A test utility class for simulating network conditions like disconnects,
 * latency, and packet loss during WebSocket E2E tests. Enables testing of
 * reconnection scenarios and network resilience.
 *
 * Features:
 * - Force disconnect clients (simulate network loss)
 * - Delay/drop messages (simulate poor network)
 * - Simulate client-side disconnect followed by reconnect
 * - Integrates with MultiClientCoordinator
 *
 * @example
 * ```typescript
 * const coordinator = new MultiClientCoordinator('http://localhost:3000');
 * const networkSimulator = new NetworkSimulator(coordinator);
 *
 * // Connect players
 * await coordinator.connect('player1', config1);
 * await coordinator.connect('player2', config2);
 *
 * // Simulate network partition for player 1
 * await networkSimulator.forceDisconnect('player1');
 *
 * // Wait some time, then reconnect
 * await networkSimulator.simulateReconnect('player1', 1000);
 * ```
 */

import type { Socket } from 'socket.io-client';
import type { MultiClientCoordinator, ClientConfig } from './MultiClientCoordinator';
import type { ServerToClientEvents, ClientToServerEvents } from '../../src/shared/types/websocket';

// ============================================================================
// Types
// ============================================================================

/**
 * Network condition configuration for simulating poor network.
 */
export interface NetworkCondition {
  /** Latency to add to messages in milliseconds */
  latencyMs?: number;
  /** Probability of dropping a message (0-1) */
  packetLoss?: number;
  /** Time in ms until automatic disconnect */
  disconnectAfter?: number;
}

/**
 * Result of message interception.
 */
export interface InterceptionResult {
  /** Action to take on the message */
  action: 'pass' | 'drop' | 'delay';
  /** Delay in ms if action is 'delay' */
  delayMs?: number;
}

/**
 * Function type for message interception.
 */
export type MessageInterceptor = (
  message: unknown,
  direction: 'inbound' | 'outbound'
) => InterceptionResult;

/**
 * State tracking for a disconnected client pending reconnection.
 */
interface DisconnectState {
  /** When the disconnect occurred */
  disconnectedAt: number;
  /** Original client configuration for reconnection */
  config: ClientConfig;
  /** The game room the client was in (if any) */
  gameId?: string;
}

/** Type for Socket.IO emit function - accepts event and variable arguments */
type SocketEmitFunction = (
  event: string,
  ...args: unknown[]
) => Socket<ServerToClientEvents, ClientToServerEvents>;

/**
 * Internal state for a client's network simulation.
 */
interface ClientNetworkState {
  /** Current network conditions applied */
  condition?: NetworkCondition;
  /** Active message interceptor */
  interceptor?: MessageInterceptor;
  /** One-shot interceptor for next message */
  nextMessageInterceptor?: MessageInterceptor;
  /** Timeout for automatic disconnect */
  disconnectTimeout?: NodeJS.Timeout;
  /** Original emit function before wrapping */
  originalEmit?: SocketEmitFunction;
  /** Whether we've wrapped the socket's emit */
  isWrapped: boolean;
}

// ============================================================================
// NetworkSimulator Class
// ============================================================================

/**
 * Simulates network conditions for testing reconnection scenarios.
 */
export class NetworkSimulator {
  private coordinator: MultiClientCoordinator;
  private clientStates: Map<string, ClientNetworkState> = new Map();
  private disconnectedClients: Map<string, DisconnectState> = new Map();

  /**
   * Creates a new NetworkSimulator.
   *
   * @param coordinator - The MultiClientCoordinator to simulate network conditions on
   */
  constructor(coordinator: MultiClientCoordinator) {
    this.coordinator = coordinator;
  }

  // ==========================================================================
  // Network Condition Management
  // ==========================================================================

  /**
   * Apply network conditions to a specific client.
   *
   * @param clientId - ID of the client
   * @param condition - Network condition to apply
   */
  setCondition(clientId: string, condition: NetworkCondition): void {
    let state = this.clientStates.get(clientId);
    if (!state) {
      state = { isWrapped: false };
      this.clientStates.set(clientId, state);
    }

    state.condition = condition;
    this.wrapSocketIfNeeded(clientId);

    // Set up automatic disconnect if specified
    if (condition.disconnectAfter !== undefined) {
      if (state.disconnectTimeout) {
        clearTimeout(state.disconnectTimeout);
      }
      state.disconnectTimeout = setTimeout(() => {
        void this.forceDisconnect(clientId);
      }, condition.disconnectAfter);
    }
  }

  /**
   * Clear network conditions for a client.
   *
   * @param clientId - ID of the client
   */
  clearCondition(clientId: string): void {
    const state = this.clientStates.get(clientId);
    if (!state) return;

    if (state.disconnectTimeout) {
      clearTimeout(state.disconnectTimeout);
      state.disconnectTimeout = undefined;
    }

    state.condition = undefined;
  }

  /**
   * Get current network condition for a client.
   *
   * @param clientId - ID of the client
   * @returns The current network condition or undefined
   */
  getCondition(clientId: string): NetworkCondition | undefined {
    return this.clientStates.get(clientId)?.condition;
  }

  // ==========================================================================
  // Force Disconnect / Reconnect
  // ==========================================================================

  /**
   * Forcibly disconnect a client (simulates network loss).
   *
   * @param clientId - ID of the client to disconnect
   * @throws Error if client not found
   */
  async forceDisconnect(clientId: string): Promise<void> {
    const socket = this.coordinator.getSocket(clientId);
    if (!socket) {
      throw new Error(`Client '${clientId}' not found or not connected`);
    }

    // Store disconnect state for potential reconnection
    const messages = this.coordinator.getMessages(clientId);
    const lastGameState = messages.filter((m) => m.eventName === 'game_state').pop();

    // Extract gameId from the last game state if available
    let gameId: string | undefined;
    if (lastGameState?.payload) {
      const payload = lastGameState.payload as { data?: { gameId?: string } };
      gameId = payload.data?.gameId;
    }

    // Get client config - we need to access the coordinator's internal state
    // Since we can't directly access it, we'll store minimal reconnection info
    this.disconnectedClients.set(clientId, {
      disconnectedAt: Date.now(),
      config: this.getClientConfigFromCoordinator(clientId),
      gameId,
    });

    // Force close without clean shutdown - use disconnect() which triggers
    // the server-side disconnect handling
    socket.disconnect();

    // Wait for disconnect to propagate
    await this.waitForDisconnection(clientId, 5000);
  }

  /**
   * Simulate reconnection after a network partition.
   *
   * @param clientId - ID of the client to reconnect
   * @param delayMs - Optional delay before reconnecting
   * @throws Error if client was not previously disconnected via forceDisconnect
   */
  async simulateReconnect(clientId: string, delayMs: number = 0): Promise<void> {
    const disconnectState = this.disconnectedClients.get(clientId);
    if (!disconnectState) {
      throw new Error(
        `Client '${clientId}' was not disconnected via forceDisconnect. ` +
          `Use forceDisconnect first or connect normally via coordinator.`
      );
    }

    if (delayMs > 0) {
      await this.delay(delayMs);
    }

    // Reconnect using stored config
    await this.coordinator.connect(clientId, disconnectState.config);

    // If the client was in a game, rejoin it
    if (disconnectState.gameId) {
      await this.coordinator.joinGame(clientId, disconnectState.gameId);
    }

    // Clean up disconnect state
    this.disconnectedClients.delete(clientId);

    // Wait for reconnection acknowledgment
    try {
      await this.coordinator.waitForEvent(clientId, 'game_state', undefined, 5000);
    } catch {
      // It's OK if we don't get a game_state immediately - the client is connected
    }
  }

  /**
   * Get the duration a client has been disconnected.
   *
   * @param clientId - ID of the client
   * @returns Duration in ms, or null if not disconnected
   */
  getDisconnectionDuration(clientId: string): number | null {
    const state = this.disconnectedClients.get(clientId);
    if (!state) return null;
    return Date.now() - state.disconnectedAt;
  }

  /**
   * Check if a client is currently disconnected (pending reconnection).
   *
   * @param clientId - ID of the client
   */
  isDisconnected(clientId: string): boolean {
    return this.disconnectedClients.has(clientId);
  }

  // ==========================================================================
  // Message Interception
  // ==========================================================================

  /**
   * Set a message interceptor for a client.
   * The interceptor is called for each message and can pass, drop, or delay it.
   *
   * @param clientId - ID of the client
   * @param interceptor - Function to intercept messages
   */
  interceptMessages(clientId: string, interceptor: MessageInterceptor): void {
    let state = this.clientStates.get(clientId);
    if (!state) {
      state = { isWrapped: false };
      this.clientStates.set(clientId, state);
    }

    state.interceptor = interceptor;
    this.wrapSocketIfNeeded(clientId);
  }

  /**
   * Clear the message interceptor for a client.
   *
   * @param clientId - ID of the client
   */
  clearInterceptor(clientId: string): void {
    const state = this.clientStates.get(clientId);
    if (state) {
      state.interceptor = undefined;
      state.nextMessageInterceptor = undefined;
    }
  }

  /**
   * Drop the next outgoing message from a client.
   *
   * @param clientId - ID of the client
   */
  dropNextMessage(clientId: string): void {
    let state = this.clientStates.get(clientId);
    if (!state) {
      state = { isWrapped: false };
      this.clientStates.set(clientId, state);
    }

    state.nextMessageInterceptor = () => {
      return { action: 'drop' };
    };
    this.wrapSocketIfNeeded(clientId);
  }

  /**
   * Delay the next outgoing message from a client.
   *
   * @param clientId - ID of the client
   * @param delayMs - Delay in milliseconds
   */
  delayNextMessage(clientId: string, delayMs: number): void {
    let state = this.clientStates.get(clientId);
    if (!state) {
      state = { isWrapped: false };
      this.clientStates.set(clientId, state);
    }

    state.nextMessageInterceptor = () => {
      return { action: 'delay', delayMs };
    };
    this.wrapSocketIfNeeded(clientId);
  }

  // ==========================================================================
  // Batch Operations
  // ==========================================================================

  /**
   * Force disconnect all connected clients.
   */
  async forceDisconnectAll(): Promise<void> {
    const clientIds = this.coordinator.getConnectedClientIds();
    await Promise.all(clientIds.map((id) => this.forceDisconnect(id)));
  }

  /**
   * Clear all network conditions for all clients.
   */
  clearAllConditions(): void {
    for (const [clientId] of this.clientStates) {
      this.clearCondition(clientId);
      this.clearInterceptor(clientId);
    }
  }

  /**
   * Clean up all state (call in afterEach).
   */
  cleanup(): void {
    // Clear all timeouts
    for (const [, state] of this.clientStates) {
      if (state.disconnectTimeout) {
        clearTimeout(state.disconnectTimeout);
      }
    }
    this.clientStates.clear();
    this.disconnectedClients.clear();
  }

  // ==========================================================================
  // Private Helpers
  // ==========================================================================

  /**
   * Wrap a client's socket emit to apply network conditions.
   */
  private wrapSocketIfNeeded(clientId: string): void {
    const state = this.clientStates.get(clientId);
    if (!state || state.isWrapped) return;

    const socket = this.coordinator.getSocket(clientId);
    if (!socket) return;

    // Store original emit
    state.originalEmit = socket.emit.bind(socket);
    state.isWrapped = true;

    // Wrap emit
    const self = this;
    (socket as any).emit = function (...args: unknown[]) {
      const event = args[0];
      const payload = args[1];

      // Apply one-shot interceptor first
      if (state?.nextMessageInterceptor) {
        const result = state.nextMessageInterceptor(payload, 'outbound');
        state.nextMessageInterceptor = undefined;

        if (result.action === 'drop') {
          return socket; // Don't send
        }
        if (result.action === 'delay' && result.delayMs) {
          setTimeout(() => {
            state.originalEmit?.apply(socket, args);
          }, result.delayMs);
          return socket;
        }
      }

      // Apply persistent interceptor
      if (state?.interceptor) {
        const result = state.interceptor(payload, 'outbound');

        if (result.action === 'drop') {
          return socket; // Don't send
        }
        if (result.action === 'delay' && result.delayMs) {
          setTimeout(() => {
            state.originalEmit?.apply(socket, args);
          }, result.delayMs);
          return socket;
        }
      }

      // Apply network conditions (latency, packet loss)
      const condition = state?.condition;
      if (condition) {
        // Packet loss
        if (condition.packetLoss !== undefined && Math.random() < condition.packetLoss) {
          return socket; // Drop message
        }

        // Latency
        if (condition.latencyMs !== undefined && condition.latencyMs > 0) {
          setTimeout(() => {
            state.originalEmit?.apply(socket, args);
          }, condition.latencyMs);
          return socket;
        }
      }

      // Normal send
      return state.originalEmit?.apply(socket, args);
    };
  }

  /**
   * Wait for a client to disconnect with timeout.
   */
  private async waitForDisconnection(clientId: string, timeoutMs: number): Promise<void> {
    const startTime = Date.now();
    while (Date.now() - startTime < timeoutMs) {
      if (!this.coordinator.isConnected(clientId)) {
        return;
      }
      await this.delay(100);
    }
    // Don't throw - the disconnect might still be processing
  }

  /**
   * Get client config from coordinator (best effort).
   * Since we can't access the coordinator's internal state directly,
   * we return a minimal config that can be used for reconnection.
   */
  private getClientConfigFromCoordinator(clientId: string): ClientConfig {
    // The coordinator stores config internally - we'll need to track it ourselves
    // For now, return a placeholder that tests should override
    return {
      playerId: clientId,
      token: '', // Tests should provide token via setClientConfig
    };
  }

  /**
   * Store client config for later reconnection.
   * Call this after coordinator.connect() to enable simulateReconnect().
   *
   * @param clientId - ID of the client
   * @param config - The client configuration used for connection
   */
  storeClientConfig(clientId: string, config: ClientConfig): void {
    // Update any existing disconnect state
    const existing = this.disconnectedClients.get(clientId);
    if (existing) {
      existing.config = config;
    }

    // Store for future disconnects
    let state = this.clientStates.get(clientId);
    if (!state) {
      state = { isWrapped: false };
      this.clientStates.set(clientId, state);
    }
    (state as any).storedConfig = config;
  }

  /**
   * Helper to delay for a specified time.
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Extended MultiClientCoordinator with Network Simulation
// ============================================================================

/**
 * Extended coordinator that includes network simulation capabilities.
 * Use this as a drop-in replacement for MultiClientCoordinator in tests
 * that need network partition simulation.
 */
export class NetworkAwareCoordinator {
  private coordinator: MultiClientCoordinator;
  private _network: NetworkSimulator;
  private clientConfigs: Map<string, ClientConfig> = new Map();

  constructor(serverUrl: string, defaultTimeout = 30000) {
    // Import dynamically to avoid circular dependency issues
    const { MultiClientCoordinator: MCC } = require('./MultiClientCoordinator');
    this.coordinator = new MCC(serverUrl, defaultTimeout);
    this._network = new NetworkSimulator(this.coordinator);
  }

  /**
   * Access the network simulator for this coordinator.
   */
  get network(): NetworkSimulator {
    return this._network;
  }

  /**
   * Connect a client and store config for reconnection.
   */
  async connect(clientId: string, config: ClientConfig): Promise<void> {
    this.clientConfigs.set(clientId, config);
    this._network.storeClientConfig(clientId, config);
    await this.coordinator.connect(clientId, config);
  }

  /**
   * Get stored config for a client.
   */
  getClientConfig(clientId: string): ClientConfig | undefined {
    return this.clientConfigs.get(clientId);
  }

  // Delegate all other methods to the underlying coordinator

  async disconnect(clientId: string): Promise<void> {
    await this.coordinator.disconnect(clientId);
  }

  async disconnectAll(): Promise<void> {
    await this.coordinator.disconnectAll();
  }

  async cleanup(): Promise<void> {
    this._network.cleanup();
    await this.coordinator.cleanup();
    this.clientConfigs.clear();
  }

  async send<E extends keyof ClientToServerEvents>(
    clientId: string,
    event: E,
    payload: Parameters<ClientToServerEvents[E]>[0]
  ): Promise<void> {
    await this.coordinator.send(clientId, event, payload);
  }

  async joinGame(clientId: string, gameId: string): Promise<void> {
    await this.coordinator.joinGame(clientId, gameId);
  }

  async leaveGame(clientId: string, gameId: string): Promise<void> {
    await this.coordinator.leaveGame(clientId, gameId);
  }

  async sendMoveById(clientId: string, gameId: string, moveId: string): Promise<void> {
    await this.coordinator.sendMoveById(clientId, gameId, moveId);
  }

  async sendChat(clientId: string, gameId: string, text: string): Promise<void> {
    await this.coordinator.sendChat(clientId, gameId, text);
  }

  async waitFor(
    clientId: string,
    condition: Parameters<MultiClientCoordinator['waitFor']>[1]
  ): Promise<unknown> {
    return await this.coordinator.waitFor(clientId, condition);
  }

  async waitForAll(
    clientIds: string[],
    condition: Parameters<MultiClientCoordinator['waitForAll']>[1]
  ): Promise<Map<string, unknown>> {
    return await this.coordinator.waitForAll(clientIds, condition);
  }

  async waitForGameState(
    clientId: string,
    predicate: Parameters<MultiClientCoordinator['waitForGameState']>[1],
    timeout?: number
  ) {
    return await this.coordinator.waitForGameState(clientId, predicate, timeout);
  }

  async waitForPhase(
    clientId: string,
    phase: Parameters<MultiClientCoordinator['waitForPhase']>[1],
    timeout?: number
  ) {
    return await this.coordinator.waitForPhase(clientId, phase, timeout);
  }

  async waitForTurn(clientId: string, playerNumber: number, timeout?: number) {
    return await this.coordinator.waitForTurn(clientId, playerNumber, timeout);
  }

  async waitForGameOver(clientId: string, timeout?: number) {
    return await this.coordinator.waitForGameOver(clientId, timeout);
  }

  async waitForEvent(
    clientId: string,
    eventName: Parameters<MultiClientCoordinator['waitForEvent']>[1],
    predicate?: Parameters<MultiClientCoordinator['waitForEvent']>[2],
    timeout?: number
  ): Promise<unknown> {
    return await this.coordinator.waitForEvent(clientId, eventName, predicate, timeout);
  }

  async executeSequence(
    actions: Parameters<MultiClientCoordinator['executeSequence']>[0]
  ): Promise<void> {
    await this.coordinator.executeSequence(actions);
  }

  async executeParallel(actions: Array<() => Promise<void>>): Promise<void> {
    await this.coordinator.executeParallel(actions);
  }

  getMessages(clientId: string) {
    return this.coordinator.getMessages(clientId);
  }

  clearMessages(clientId: string): void {
    this.coordinator.clearMessages(clientId);
  }

  getMessagesMatching(
    clientId: string,
    filter: Parameters<MultiClientCoordinator['getMessagesMatching']>[1]
  ) {
    return this.coordinator.getMessagesMatching(clientId, filter);
  }

  getLastGameState(clientId: string) {
    return this.coordinator.getLastGameState(clientId);
  }

  getConnectedClientIds(): string[] {
    return this.coordinator.getConnectedClientIds();
  }

  isConnected(clientId: string): boolean {
    return this.coordinator.isConnected(clientId);
  }

  getSocket(clientId: string) {
    return this.coordinator.getSocket(clientId);
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Creates a new NetworkSimulator for an existing coordinator.
 */
export function createNetworkSimulator(coordinator: MultiClientCoordinator): NetworkSimulator {
  return new NetworkSimulator(coordinator);
}

/**
 * Creates a new NetworkAwareCoordinator with built-in network simulation.
 */
export function createNetworkAwareCoordinator(
  serverUrl: string,
  defaultTimeout?: number
): NetworkAwareCoordinator {
  return new NetworkAwareCoordinator(serverUrl, defaultTimeout);
}
