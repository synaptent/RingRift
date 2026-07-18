/**
 * ClockManager - Player Timer Management
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Manages per-player move timers for enforcing time controls in games.
 * Extracted from GameEngine to improve separation of concerns.
 *
 * Features:
 * - Start/stop individual player timers
 * - Automatic timeout callback when time expires
 * - Clean disposal of all timers
 * - Test environment awareness (no-op in Jest)
 *
 * @module ClockManager
 */

import { isJestRuntime } from '../../../shared/utils/envFlags';

type TimerHandle = ReturnType<typeof setTimeout>;

export interface ClockManagerConfig {
  /** Called when a player's time expires */
  onTimeout: (playerNumber: number) => void;
  /** If true, timers are disabled (for test environments) */
  disableTimers?: boolean;
}

export interface PlayerTimeInfo {
  playerNumber: number;
  timeRemaining: number;
  isAI: boolean;
}

/**
 * Manages move timers for enforcing time controls.
 *
 * In Jest test environments, timers are no-oped to prevent keeping the
 * Node event loop alive after tests complete.
 */
export class ClockManager {
  private timers: Map<number, TimerHandle> = new Map();
  private turnStartTimes: Map<number, number> = new Map(); // Track when each player's turn started
  private config: ClockManagerConfig;
  private isDisabled: boolean;

  constructor(clockConfig: ClockManagerConfig) {
    this.config = clockConfig;
    // Disable timers only inside Jest workers. Browser E2E servers also run
    // with NODE_ENV=test, but must exercise production-like clock expiry.
    this.isDisabled = clockConfig.disableTimers ?? isJestRuntime();
  }

  /**
   * Start a timer for the specified player.
   *
   * @param player - Player info including time remaining
   */
  startTimer(player: PlayerTimeInfo): void {
    // Don't start timers in test/disabled mode
    if (this.isDisabled) {
      return;
    }

    // Clear any existing timer for this player
    this.stopTimer(player.playerNumber);

    const timer = setTimeout(() => {
      this.config.onTimeout(player.playerNumber);
    }, player.timeRemaining);

    this.timers.set(player.playerNumber, timer);
    this.turnStartTimes.set(player.playerNumber, Date.now());
  }

  /**
   * Stop the timer for the specified player.
   *
   * @param playerNumber - The player whose timer to stop
   */
  stopTimer(playerNumber: number): void {
    const timer = this.timers.get(playerNumber);
    if (timer) {
      clearTimeout(timer);
      this.timers.delete(playerNumber);
    }
    this.turnStartTimes.delete(playerNumber);
  }

  /**
   * Stop the timer and return the elapsed time in milliseconds.
   * Returns 0 if the timer was not running or in disabled mode.
   *
   * @param playerNumber - The player whose timer to stop
   * @returns Elapsed time in milliseconds
   */
  stopTimerAndGetElapsed(playerNumber: number): number {
    const startTime = this.turnStartTimes.get(playerNumber);
    const elapsed = startTime ? Date.now() - startTime : 0;
    this.stopTimer(playerNumber);
    return elapsed;
  }

  /**
   * Stop all active timers.
   * Call this when the game ends.
   */
  stopAllTimers(): void {
    for (const timer of this.timers.values()) {
      clearTimeout(timer);
    }
    this.timers.clear();
    this.turnStartTimes.clear();
  }

  /**
   * Check if a timer is active for the specified player.
   *
   * @param playerNumber - The player to check
   * @returns true if a timer is currently running
   */
  hasActiveTimer(playerNumber: number): boolean {
    return this.timers.has(playerNumber);
  }

  /**
   * Get the number of active timers.
   * Useful for debugging and testing.
   */
  get activeTimerCount(): number {
    return this.timers.size;
  }
}
