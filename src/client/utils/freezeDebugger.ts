/**
 * Freeze Debugger - Browser-side instrumentation to capture game state before freezes
 *
 * The problem: When the browser freezes, you can't access DevTools or export state.
 * The solution: Capture state BEFORE each AI turn, so the last saved state is the
 * problematic one that caused the freeze.
 *
 * How it works:
 * 1. Before each AI turn, save state to localStorage (fast, sync)
 * 2. Set a watchdog timer - if turn exceeds threshold, log warning
 * 3. Periodically save to IndexedDB for larger history
 * 4. After freeze, open new tab and run: FreezeDebugger.getLastState()
 *
 * Usage:
 *   // Enable in browser console:
 *   window.__FREEZE_DEBUGGER__.enable();
 *
 *   // After a freeze, open a new tab to the same origin and run:
 *   window.__FREEZE_DEBUGGER__.getLastState();
 *
 *   // Or directly access localStorage:
 *   JSON.parse(localStorage.getItem('ringrift_freeze_debug_state'));
 */

import { GameState } from '../../shared/types/game';
import { hashGameState } from '../../shared/engine/core';
import { serializeGameState } from '../../shared/engine/contracts/serialization';

// Configuration
const WATCHDOG_WARNING_MS = 2000; // Warn after 2 seconds
const WATCHDOG_CRITICAL_MS = 5000; // Critical after 5 seconds
const MAX_HISTORY_SIZE = 10; // Keep last 10 states in memory
const LOCALSTORAGE_KEY = 'ringrift_freeze_debug_state';
const LOCALSTORAGE_HISTORY_KEY = 'ringrift_freeze_debug_history';
const INDEXEDDB_NAME = 'ringrift_freeze_debug';
const INDEXEDDB_STORE = 'states';

interface FreezeDebugState {
  timestamp: string;
  turnNumber: number;
  boardType: string;
  numPlayers: number;
  currentPhase: string;
  currentPlayer: number;
  stackCount: number;
  markerCount: number;
  stateHash: string;
  serializedState: ReturnType<typeof serializeGameState>;
  rawState?: GameState;
}

interface FreezeDebugHistory {
  states: FreezeDebugState[];
  lastUpdated: string;
}

class FreezeDebuggerClass {
  private enabled = false;
  private turnNumber = 0;
  private watchdogTimer: ReturnType<typeof setTimeout> | null = null;
  private criticalTimer: ReturnType<typeof setTimeout> | null = null;
  private currentState: FreezeDebugState | null = null;
  private history: FreezeDebugState[] = [];
  private db: IDBDatabase | null = null;

  constructor() {
    // Auto-enable in development
    if (typeof window !== 'undefined') {
      // Expose globally for console access
      (window as unknown as { __FREEZE_DEBUGGER__: FreezeDebugger }).__FREEZE_DEBUGGER__ = this;

      // Try to open IndexedDB
      this.initIndexedDB();

      // Check for auto-enable flag
      if (localStorage.getItem('ringrift_freeze_debug_enabled') === 'true') {
        this.enable();
      }
    }
  }

  /**
   * Enable freeze debugging
   */
  enable(): void {
    this.enabled = true;
    localStorage.setItem('ringrift_freeze_debug_enabled', 'true');
    console.log(
      '%c🔍 Freeze Debugger ENABLED',
      'background: #2563eb; color: white; padding: 4px 8px; border-radius: 4px;'
    );
    console.log('State will be saved before each AI turn.');
    console.log('After a freeze, open a new tab and run: __FREEZE_DEBUGGER__.getLastState()');
  }

  /**
   * Disable freeze debugging
   */
  disable(): void {
    this.enabled = false;
    localStorage.removeItem('ringrift_freeze_debug_enabled');
    this.clearWatchdog();
    console.log('🔍 Freeze Debugger disabled');
  }

  /**
   * Check if debugging is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Call this BEFORE each AI turn starts
   */
  beforeAITurn(state: GameState, turnNumber?: number): void {
    if (!this.enabled) return;

    this.turnNumber = turnNumber ?? this.turnNumber + 1;

    try {
      // Capture state
      this.currentState = {
        timestamp: new Date().toISOString(),
        turnNumber: this.turnNumber,
        boardType: state.boardType,
        numPlayers: state.players.length,
        currentPhase: state.currentPhase,
        currentPlayer: state.currentPlayer,
        stackCount: state.board.stacks.size,
        markerCount: state.board.markers.size,
        stateHash: hashGameState(state),
        serializedState: serializeGameState(state),
        // Don't include rawState in localStorage (too large), only in IndexedDB
      };

      // Save to localStorage immediately (sync, fast)
      this.saveToLocalStorage(this.currentState);

      // Add to history
      this.history.push(this.currentState);
      if (this.history.length > MAX_HISTORY_SIZE) {
        this.history.shift();
      }

      // Save history to localStorage
      this.saveHistoryToLocalStorage();

      // Save to IndexedDB (async, can include full state)
      this.saveToIndexedDB({ ...this.currentState, rawState: state });

      // Start watchdog timer
      this.startWatchdog();

      // Log progress periodically
      if (this.turnNumber % 50 === 0) {
        console.log(
          `🔍 Turn ${this.turnNumber}: ${state.currentPhase}, stacks=${state.board.stacks.size}`
        );
      }
    } catch (err) {
      console.error('Freeze debugger error in beforeAITurn:', err);
    }
  }

  /**
   * Call this AFTER each AI turn completes
   */
  afterAITurn(): void {
    if (!this.enabled) return;
    this.clearWatchdog();
  }

  /**
   * Get the last captured state (call from a new tab after freeze)
   */
  getLastState(): FreezeDebugState | null {
    try {
      const stored = localStorage.getItem(LOCALSTORAGE_KEY);
      if (stored) {
        const state = JSON.parse(stored) as FreezeDebugState;
        console.log('%c📦 Last captured state before freeze:', 'font-weight: bold;');
        console.log(`  Turn: ${state.turnNumber}`);
        console.log(`  Board: ${state.boardType} (${state.numPlayers} players)`);
        console.log(`  Phase: ${state.currentPhase}, Player: ${state.currentPlayer}`);
        console.log(`  Stacks: ${state.stackCount}, Markers: ${state.markerCount}`);
        console.log(`  Hash: ${state.stateHash}`);
        console.log(`  Captured at: ${state.timestamp}`);
        console.log('\nFull state available via: __FREEZE_DEBUGGER__.exportState()');
        return state;
      }
    } catch (err) {
      console.error('Error reading last state:', err);
    }
    console.log('No freeze debug state found');
    return null;
  }

  /**
   * Get recent state history
   */
  getHistory(): FreezeDebugState[] {
    try {
      const stored = localStorage.getItem(LOCALSTORAGE_HISTORY_KEY);
      if (stored) {
        const history = JSON.parse(stored) as FreezeDebugHistory;
        console.log(`📜 Found ${history.states.length} states in history`);
        return history.states;
      }
    } catch (err) {
      console.error('Error reading history:', err);
    }
    return [];
  }

  /**
   * Export the last state as a downloadable JSON file
   */
  exportState(): void {
    const state = this.getLastState();
    if (!state) {
      console.log('No state to export');
      return;
    }

    const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `freeze-state-${state.boardType}-turn${state.turnNumber}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log('State exported to file');
  }

  /**
   * Export all history as a downloadable JSON file
   */
  exportHistory(): void {
    const history = this.getHistory();
    if (history.length === 0) {
      console.log('No history to export');
      return;
    }

    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `freeze-history-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`Exported ${history.length} states to file`);
  }

  /**
   * Clear all stored debug data
   */
  clearData(): void {
    localStorage.removeItem(LOCALSTORAGE_KEY);
    localStorage.removeItem(LOCALSTORAGE_HISTORY_KEY);
    this.history = [];
    this.currentState = null;

    // Clear IndexedDB
    if (this.db) {
      const tx = this.db.transaction([INDEXEDDB_STORE], 'readwrite');
      tx.objectStore(INDEXEDDB_STORE).clear();
    }

    console.log('Freeze debug data cleared');
  }

  /**
   * Get stats about captured data
   */
  getStats(): void {
    const lastState = this.getLastState();
    const history = this.getHistory();

    console.log('%c📊 Freeze Debugger Stats', 'font-weight: bold;');
    console.log(`  Enabled: ${this.enabled}`);
    console.log(`  Current turn: ${this.turnNumber}`);
    console.log(`  States in memory: ${this.history.length}`);
    console.log(`  States in localStorage history: ${history.length}`);
    console.log(`  Last state captured: ${lastState?.timestamp ?? 'none'}`);
  }

  // Private methods

  private saveToLocalStorage(state: FreezeDebugState): void {
    try {
      // Don't include rawState in localStorage (too large)
      const { rawState: _rawState, ...stateWithoutRaw } = state as FreezeDebugState & {
        rawState?: GameState;
      };
      localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(stateWithoutRaw));
    } catch (err) {
      // localStorage might be full
      console.warn('Failed to save to localStorage:', err);
    }
  }

  private saveHistoryToLocalStorage(): void {
    try {
      const historyData: FreezeDebugHistory = {
        states: this.history.map(({ rawState: _rawState, ...rest }) => rest),
        lastUpdated: new Date().toISOString(),
      };
      localStorage.setItem(LOCALSTORAGE_HISTORY_KEY, JSON.stringify(historyData));
    } catch (_err) {
      // localStorage might be full - trim history
      if (this.history.length > 5) {
        this.history = this.history.slice(-5);
        this.saveHistoryToLocalStorage();
      }
    }
  }

  private async initIndexedDB(): Promise<void> {
    if (typeof indexedDB === 'undefined') return;

    try {
      const request = indexedDB.open(INDEXEDDB_NAME, 1);

      request.onerror = () => {
        console.warn('IndexedDB not available for freeze debugger');
      };

      request.onsuccess = () => {
        this.db = request.result;
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(INDEXEDDB_STORE)) {
          db.createObjectStore(INDEXEDDB_STORE, { keyPath: 'timestamp' });
        }
      };
    } catch (err) {
      console.warn('Failed to init IndexedDB:', err);
    }
  }

  private saveToIndexedDB(state: FreezeDebugState & { rawState?: GameState }): void {
    if (!this.db) return;

    try {
      const tx = this.db.transaction([INDEXEDDB_STORE], 'readwrite');
      const store = tx.objectStore(INDEXEDDB_STORE);
      store.put(state);

      // Keep only last 100 entries
      const countRequest = store.count();
      countRequest.onsuccess = () => {
        if (countRequest.result > 100) {
          const cursorRequest = store.openCursor();
          let deleteCount = countRequest.result - 100;
          cursorRequest.onsuccess = (event) => {
            const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
            if (cursor && deleteCount > 0) {
              cursor.delete();
              deleteCount--;
              cursor.continue();
            }
          };
        }
      };
    } catch (_err) {
      // Ignore IndexedDB errors - localStorage is the primary backup
    }
  }

  private startWatchdog(): void {
    this.clearWatchdog();

    // Warning timer
    this.watchdogTimer = setTimeout(() => {
      console.warn(
        `%c⚠️ AI turn ${this.turnNumber} taking >${WATCHDOG_WARNING_MS}ms`,
        'background: #f59e0b; color: black; padding: 2px 6px; border-radius: 4px;'
      );
      console.warn('State has been saved. If browser freezes, open new tab and run:');
      console.warn('  __FREEZE_DEBUGGER__.getLastState()');
    }, WATCHDOG_WARNING_MS);

    // Critical timer
    this.criticalTimer = setTimeout(() => {
      console.error(
        `%c🔴 AI turn ${this.turnNumber} CRITICAL: >${WATCHDOG_CRITICAL_MS}ms - possible freeze!`,
        'background: #dc2626; color: white; padding: 2px 6px; border-radius: 4px;'
      );
      console.error('State saved to localStorage. If browser becomes unresponsive:');
      console.error('1. Force quit browser');
      console.error('2. Reopen and navigate to same origin');
      console.error('3. Run: __FREEZE_DEBUGGER__.getLastState()');
      console.error('4. Run: __FREEZE_DEBUGGER__.exportState()');
    }, WATCHDOG_CRITICAL_MS);
  }

  private clearWatchdog(): void {
    if (this.watchdogTimer) {
      clearTimeout(this.watchdogTimer);
      this.watchdogTimer = null;
    }
    if (this.criticalTimer) {
      clearTimeout(this.criticalTimer);
      this.criticalTimer = null;
    }
  }
}

// Singleton instance
export const FreezeDebugger = new FreezeDebuggerClass();

// Export for use in other modules
export default FreezeDebugger;
