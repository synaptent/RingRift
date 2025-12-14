/**
 * Setup file for jsdom environment
 * Runs BEFORE test framework is installed
 */

// Silence dotenv v17+ noisy startup logs in Jest runs.
process.env.DOTENV_CONFIG_QUIET = 'true';

// Default test env + quiet logging unless a test overrides explicitly.
process.env.NODE_ENV = process.env.NODE_ENV || 'test';
process.env.LOG_LEVEL = process.env.LOG_LEVEL || 'error';

import { MessageChannel as NodeMessageChannel } from 'worker_threads';

// ═══════════════════════════════════════════════════════════════════════════
// structuredClone Polyfill for Node.js 16 Compatibility
// ═══════════════════════════════════════════════════════════════════════════
// structuredClone is available in Node 17+, but not in Node 16.
// This polyfill uses JSON serialization as a fallback for basic object cloning.
// For more complex objects (with circular refs, Map, Set, etc.), the polyfill
// will fail, but for typical game state objects it works.
if (typeof globalThis.structuredClone === 'undefined') {
  (globalThis as any).structuredClone = <T>(obj: T): T => {
    return JSON.parse(JSON.stringify(obj));
  };
}

// Ensure MessageChannel is available for React 18+/19 scheduling logic in
// environments where jsdom does not provide it by default. We alias Node's
// worker_threads implementation onto the global object so that React's
// server/browser bundles (used by react-dom/server) can rely on it.
if (typeof (globalThis as any).MessageChannel === 'undefined') {
  (globalThis as any).MessageChannel = NodeMessageChannel as unknown as typeof MessageChannel;
}

// Mock Vite env for client code that relies on import.meta.env. In the real
// bundle, vite.config.ts defines `globalThis.__VITE_ENV__ = import.meta.env`.
// Here we provide a minimal test-friendly subset and also expose it via a
// synthetic global.importMeta for any code that still references import.meta.
const viteEnv = {
  MODE: 'test',
  DEV: false,
  PROD: false,
  SSR: false,
  VITE_API_URL: 'http://localhost:3000',
  VITE_WS_URL: 'http://localhost:3000',
};

Object.defineProperty(global, 'importMeta', {
  value: {
    env: viteEnv,
  },
  writable: true,
  configurable: true,
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).__VITE_ENV__ = viteEnv;

// ═══════════════════════════════════════════════════════════════════════════
// WebSocket Mock for Client-Side Tests
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Mock WebSocket implementation for testing.
 *
 * Supports standard WebSocket properties and methods, plus helper methods
 * for simulating server behavior in tests.
 *
 * Usage in tests:
 * ```typescript
 * const ws = new WebSocket('ws://localhost:3000');
 *
 * // Simulate receiving a message
 * (ws as MockWebSocket)._simulateMessage(JSON.stringify({ type: 'update' }));
 *
 * // Simulate an error
 * (ws as MockWebSocket)._simulateError();
 *
 * // Simulate server closing connection
 * (ws as MockWebSocket)._simulateClose(1000, 'Normal closure');
 * ```
 */
class MockWebSocket {
  // WebSocket ready state constants
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  // Instance ready state constants (for compatibility)
  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSING = 2;
  readonly CLOSED = 3;

  url: string;
  readyState: number;
  protocol: string = '';
  extensions: string = '';
  bufferedAmount: number = 0;
  binaryType: BinaryType = 'blob';

  // Event handlers
  onopen: ((this: WebSocket, ev: Event) => any) | null = null;
  onclose: ((this: WebSocket, ev: CloseEvent) => any) | null = null;
  onmessage: ((this: WebSocket, ev: MessageEvent) => any) | null = null;
  onerror: ((this: WebSocket, ev: Event) => any) | null = null;

  // Track sent messages for test assertions
  private _sentMessages: (string | ArrayBuffer | Blob | ArrayBufferView)[] = [];
  private _eventListeners: Map<string, Set<EventListener>> = new Map();

  constructor(url: string | URL, protocols?: string | string[]) {
    this.url = typeof url === 'string' ? url : url.toString();
    this.readyState = MockWebSocket.CONNECTING;

    if (protocols) {
      this.protocol = Array.isArray(protocols) ? protocols[0] || '' : protocols;
    }

    // Auto-open in next tick for testing convenience
    // This simulates successful connection
    setTimeout(() => {
      if (this.readyState === MockWebSocket.CONNECTING) {
        this.readyState = MockWebSocket.OPEN;
        const openEvent = new Event('open');
        this._dispatchEvent('open', openEvent);
        this.onopen?.call(this as unknown as WebSocket, openEvent);
      }
    }, 0);
  }

  /**
   * Send data through the WebSocket
   */
  send(data: string | ArrayBuffer | Blob | ArrayBufferView): void {
    if (this.readyState === MockWebSocket.CONNECTING) {
      throw new DOMException(
        "Failed to execute 'send' on 'WebSocket': Still in CONNECTING state.",
        'InvalidStateError'
      );
    }
    if (this.readyState !== MockWebSocket.OPEN) {
      // WebSocket spec says to silently discard if not OPEN
      return;
    }
    this._sentMessages.push(data);
  }

  /**
   * Close the WebSocket connection
   */
  close(code: number = 1000, reason: string = ''): void {
    if (this.readyState === MockWebSocket.CLOSING || this.readyState === MockWebSocket.CLOSED) {
      return;
    }

    this.readyState = MockWebSocket.CLOSING;

    // Simulate async close
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED;
      const closeEvent = new CloseEvent('close', {
        code,
        reason,
        wasClean: code === 1000,
      });
      this._dispatchEvent('close', closeEvent);
      this.onclose?.call(this as unknown as WebSocket, closeEvent);
    }, 0);
  }

  /**
   * Add event listener
   */
  addEventListener(type: string, listener: EventListener): void {
    if (!this._eventListeners.has(type)) {
      this._eventListeners.set(type, new Set());
    }
    this._eventListeners.get(type)!.add(listener);
  }

  /**
   * Remove event listener
   */
  removeEventListener(type: string, listener: EventListener): void {
    this._eventListeners.get(type)?.delete(listener);
  }

  /**
   * Dispatch event to listeners
   */
  dispatchEvent(event: Event): boolean {
    this._dispatchEvent(event.type, event);
    return true;
  }

  private _dispatchEvent(type: string, event: Event): void {
    const listeners = this._eventListeners.get(type);
    if (listeners) {
      listeners.forEach((listener) => {
        try {
          listener.call(this, event);
        } catch (e) {
          console.error('Error in WebSocket event listener:', e);
        }
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Test Helper Methods
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Simulate receiving a message from the server
   */
  _simulateMessage(data: string | ArrayBuffer | Blob): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      console.warn('Cannot simulate message on non-open WebSocket');
      return;
    }
    const messageEvent = new MessageEvent('message', { data });
    this._dispatchEvent('message', messageEvent);
    this.onmessage?.call(this as unknown as WebSocket, messageEvent);
  }

  /**
   * Simulate a connection error
   */
  _simulateError(message?: string): void {
    const errorEvent = new Event('error');
    // Add message property for debugging if provided
    if (message) {
      Object.defineProperty(errorEvent, 'message', { value: message });
    }
    this._dispatchEvent('error', errorEvent);
    this.onerror?.call(this as unknown as WebSocket, errorEvent);
  }

  /**
   * Simulate server closing the connection
   */
  _simulateClose(code: number = 1000, reason: string = ''): void {
    if (this.readyState === MockWebSocket.CLOSED) {
      return;
    }
    this.readyState = MockWebSocket.CLOSED;
    const closeEvent = new CloseEvent('close', {
      code,
      reason,
      wasClean: code === 1000,
    });
    this._dispatchEvent('close', closeEvent);
    this.onclose?.call(this as unknown as WebSocket, closeEvent);
  }

  /**
   * Simulate successful connection opening
   * Useful when you want to control when connection opens
   */
  _simulateOpen(): void {
    if (this.readyState !== MockWebSocket.CONNECTING) {
      console.warn('Cannot simulate open on non-connecting WebSocket');
      return;
    }
    this.readyState = MockWebSocket.OPEN;
    const openEvent = new Event('open');
    this._dispatchEvent('open', openEvent);
    this.onopen?.call(this as unknown as WebSocket, openEvent);
  }

  /**
   * Get all messages that have been sent through this WebSocket
   */
  _getSentMessages(): (string | ArrayBuffer | Blob | ArrayBufferView)[] {
    return [...this._sentMessages];
  }

  /**
   * Clear the sent messages log
   */
  _clearSentMessages(): void {
    this._sentMessages = [];
  }

  /**
   * Get the last sent message
   */
  _getLastSentMessage(): string | ArrayBuffer | Blob | ArrayBufferView | undefined {
    return this._sentMessages[this._sentMessages.length - 1];
  }

  /**
   * Set ready state manually (for edge case testing)
   */
  _setReadyState(state: number): void {
    this.readyState = state;
  }
}

// Export the MockWebSocket class for type usage in tests
export { MockWebSocket };

// Add to global for browser environment simulation
(global as any).WebSocket = MockWebSocket;

// Also set on globalThis for modern environments
if (typeof globalThis !== 'undefined') {
  (globalThis as any).WebSocket = MockWebSocket;
}
