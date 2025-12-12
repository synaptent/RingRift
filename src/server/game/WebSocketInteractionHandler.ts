import { Server as SocketIOServer } from 'socket.io';
import { PlayerChoice, PlayerChoiceResponse } from '../../shared/types/game';
import { PlayerInteractionHandler } from './PlayerInteractionManager';
import { logger } from '../utils/logger';
import {
  type ChoiceStatus,
  makePendingChoiceStatus,
  markChoiceCanceled,
  markChoiceExpired,
  markChoiceFulfilled,
  markChoiceRejected,
} from '../../shared/stateMachines/choice';

interface PendingChoice {
  choice: PlayerChoice;
  resolve: (response: PlayerChoiceResponse<unknown>) => void;
  reject: (err: Error) => void;
  timeoutHandle: NodeJS.Timeout;
  status: ChoiceStatus;
}

/**
 * WebSocketInteractionHandler bridges PlayerInteractionManager to Socket.IO.
 *
 * It is intentionally transport-focused and does not know about Express,
 * Prisma, or any HTTP concerns. Its responsibilities are:
 * - Emit `player_choice_required` events to the appropriate client(s)
 * - Track pending choices keyed by gameId + playerNumber + choiceId
 * - Resolve/reject Promises when `player_choice_response` arrives or a timeout fires
 * - Perform basic server-side validation of the selected option
 */
export class WebSocketInteractionHandler implements PlayerInteractionHandler {
  private readonly pending = new Map<string, PendingChoice>();

  /**
   * Diagnostic record of the last known lifecycle state for each choice,
   * keyed by `${gameId}:${playerNumber}:${choiceId}`. This is primarily used
   * in tests and incident debugging; it does not affect gameplay.
   */
  private readonly lastChoiceStatuses = new Map<string, ChoiceStatus>();

  constructor(
    private readonly io: SocketIOServer,
    private readonly gameId: string,
    /**
     * Resolve a numeric playerNumber (1..N) to some Socket.IO target.
     * For now this is typically a socket id, but it could be a room name.
     */
    private readonly getTargetForPlayer: (playerNumber: number) => string | undefined,
    private readonly defaultTimeoutMs: number = 30_000
  ) {}

  /**
   * Core interface required by PlayerInteractionHandler.
   *
   * GameEngine calls this via PlayerInteractionManager.requestChoice().
   */
  async requestChoice(choice: PlayerChoice): Promise<PlayerChoiceResponse<unknown>> {
    const key = this.getKey(choice.gameId, choice.id, choice.playerNumber);

    if (this.pending.has(key)) {
      throw new Error(`WebSocketInteractionHandler: choice already pending for key ${key}`);
    }

    const target = this.getTargetForPlayer(choice.playerNumber);
    if (!target) {
      throw new Error(
        `WebSocketInteractionHandler: no WebSocket target for player ${choice.playerNumber} in game ${choice.gameId}`
      );
    }

    const timeoutMs = choice.timeoutMs ?? this.defaultTimeoutMs;

    logger.info('WebSocketInteractionHandler: emitting player_choice_required', {
      gameId: choice.gameId,
      playerNumber: choice.playerNumber,
      choiceId: choice.id,
      type: choice.type,
      timeoutMs,
    });

    const initialStatus = makePendingChoiceStatus(choice, timeoutMs);
    this.lastChoiceStatuses.set(key, initialStatus);

    return new Promise<PlayerChoiceResponse<unknown>>((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        const pending = this.pending.get(key);
        if (!pending) {
          return;
        }

        const expiredStatus = markChoiceExpired(pending.status);
        this.lastChoiceStatuses.set(key, expiredStatus);
        this.pending.delete(key);

        const err = new Error(
          `Player choice timed out after ${timeoutMs}ms (choiceId=${choice.id})`
        );
        logger.warn('WebSocketInteractionHandler: choice timeout', {
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
          type: choice.type,
        });
        reject(err);
      }, timeoutMs);

      this.pending.set(key, {
        choice,
        resolve,
        reject,
        timeoutHandle,
        status: initialStatus,
      });

      // Emit the choice to the client. This can target a specific socket id,
      // a per-user room, or any other Socket.IO namespace configured by
      // WebSocketServer.
      this.io.to(target).emit('player_choice_required', choice);
    });
  }

  /**
   * Called from WebSocketServer when a client sends `player_choice_response`.
   *
   * This method:
   * - Locates the matching pending choice
   * - Validates the selectedOption against choice.options
   * - Resolves or rejects the corresponding Promise
   */
  handleChoiceResponse(response: PlayerChoiceResponse<unknown>): void {
    const { choiceId, playerNumber } = response;
    let key = this.getKey(this.gameId, choiceId, playerNumber);
    let pending = this.pending.get(key);

    if (!pending) {
      // Fallback: locate any pending choice with the same gameId + choiceId,
      // regardless of playerNumber. This allows us to reject mismatched
      // responses instead of leaving choices dangling indefinitely.
      for (const [pendingKey, value] of this.pending.entries()) {
        const parts = pendingKey.split(':');
        if (parts.length === 3) {
          const [gId, _pNum, cId] = parts;
          if (gId === this.gameId && cId === choiceId) {
            key = pendingKey;
            pending = value;
            break;
          }
        }
      }

      if (!pending) {
        logger.warn('WebSocketInteractionHandler: no pending choice for response', {
          gameId: this.gameId,
          choiceId,
          playerNumber,
        });
        return;
      }
    }

    const { choice, resolve, reject, timeoutHandle, status } = pending;

    // Optional assertion: log a warning if the response's choiceType
    // is present and does not match the original choice.type. This is
    // non-fatal and exists primarily as a diagnostic aid while the
    // choice system is still being integrated across transports.
    if (response.choiceType && response.choiceType !== choice.type) {
      logger.warn('WebSocketInteractionHandler: choiceType mismatch', {
        gameId: this.gameId,
        choiceId,
        expectedType: choice.type,
        actualType: response.choiceType,
      });
    }

    // Basic sanity check: the same player must answer the choice.
    if (playerNumber !== choice.playerNumber) {
      clearTimeout(timeoutHandle);
      this.pending.delete(key);

      const rejectedStatus = markChoiceRejected(status, 'PLAYER_MISMATCH');
      this.lastChoiceStatuses.set(key, rejectedStatus);

      const err = new Error(
        `playerNumber mismatch for choice ${choiceId}: expected ${choice.playerNumber}, got ${playerNumber}`
      );
      logger.warn('WebSocketInteractionHandler: playerNumber mismatch', {
        gameId: this.gameId,
        choiceId,
        expectedPlayer: choice.playerNumber,
        actualPlayer: playerNumber,
      });
      return reject(err);
    }

    // Server-side validation: ensure selectedOption is one of choice.options.
    if (!this.isValidSelectedOption(choice, response.selectedOption)) {
      clearTimeout(timeoutHandle);
      this.pending.delete(key);

      const rejectedStatus = markChoiceRejected(status, 'INVALID_OPTION');
      this.lastChoiceStatuses.set(key, rejectedStatus);

      const err = new Error(`Invalid selectedOption for choice ${choiceId} (type=${choice.type})`);
      logger.warn('WebSocketInteractionHandler: invalid selectedOption', {
        gameId: this.gameId,
        choiceId,
        playerNumber,
        type: choice.type,
      });
      return reject(err);
    }

    clearTimeout(timeoutHandle);
    this.pending.delete(key);

    const fulfilledStatus = markChoiceFulfilled(status);
    this.lastChoiceStatuses.set(key, fulfilledStatus);

    resolve(response);
  }

  /**
   * Optional: allow cancellation from server side (e.g., game ended).
   * This will reject the pending Promise and notify clients so they can
   * clear any UI related to this choice.
   */
  cancelChoice(gameId: string, choiceId: string, playerNumber: number): void {
    const key = this.getKey(gameId, choiceId, playerNumber);
    const pending = this.pending.get(key);
    if (!pending) return;

    const { timeoutHandle, reject, status } = pending;
    clearTimeout(timeoutHandle);
    this.pending.delete(key);

    const canceledStatus = markChoiceCanceled(status, 'SERVER_CANCEL');
    this.lastChoiceStatuses.set(key, canceledStatus);

    reject(new Error(`Choice ${choiceId} was cancelled by server`));

    logger.info('WebSocketInteractionHandler: choice cancelled', {
      gameId,
      choiceId,
      playerNumber,
    });

    // Notify all clients in the game room so they can clear any UI.
    this.io.to(gameId).emit('player_choice_canceled', choiceId);
  }

  /**
   * Cancel all pending choices for a specific player.
   * Called when a player disconnects and the reconnection window expires.
   * This ensures stale choices don't block game progression.
   */
  cancelAllChoicesForPlayer(playerNumber: number): void {
    const keysToCancel: string[] = [];

    for (const [key, pending] of this.pending.entries()) {
      if (pending.choice.playerNumber === playerNumber && pending.choice.gameId === this.gameId) {
        keysToCancel.push(key);
      }
    }

    for (const key of keysToCancel) {
      const pending = this.pending.get(key);
      if (!pending) continue;

      const { choice, timeoutHandle, reject, status } = pending;
      clearTimeout(timeoutHandle);
      this.pending.delete(key);

      const canceledStatus = markChoiceCanceled(status, 'DISCONNECT');
      this.lastChoiceStatuses.set(key, canceledStatus);

      reject(new Error(`Choice ${choice.id} was cancelled due to player disconnect`));

      logger.info('WebSocketInteractionHandler: choice cancelled due to disconnect', {
        gameId: this.gameId,
        choiceId: choice.id,
        playerNumber,
      });

      // Notify all clients in the game room so they can clear any UI.
      this.io.to(this.gameId).emit('player_choice_canceled', choice.id);
    }
  }

  /**
   * Cancel all pending choices for this game.
   *
   * Intended for terminal flows (e.g., game_over, resign, abandonment) so that
   * no in-flight decision promises or client UI remain after completion.
   */
  cancelAllChoices(): void {
    const pendings = Array.from(this.pending.values());
    for (const pending of pendings) {
      const { choice } = pending;
      this.cancelChoice(choice.gameId, choice.id, choice.playerNumber);
    }
  }

  /**
   * Get count of pending choices (useful for testing and diagnostics)
   */
  getPendingCount(): number {
    return this.pending.size;
  }

  /**
   * Expose a narrow diagnostic view for tests so they can assert the
   * lifecycle of individual choices without reaching into private maps.
   */
  public getLastChoiceStatusSnapshotForTesting(
    choiceId: string,
    playerNumber: number
  ): ChoiceStatus | undefined {
    const key = this.getKey(this.gameId, choiceId, playerNumber);
    return this.lastChoiceStatuses.get(key);
  }

  private getKey(gameId: string, choiceId: string, playerNumber: number): string {
    return `${gameId}:${playerNumber}:${choiceId}`;
  }

  private isValidSelectedOption(choice: PlayerChoice, selectedOption: unknown): boolean {
    // All PlayerChoice variants have an `options` array. We use a simple
    // structural comparison for now; this can be upgraded to per-type
    // discrimination if needed.
    const options = (choice as { options?: readonly unknown[] }).options;
    if (!options || !Array.isArray(options)) {
      return false;
    }

    return options.some((opt) => this.shallowOptionEquals(opt, selectedOption));
  }

  private shallowOptionEquals(a: unknown, b: unknown): boolean {
    // For now, rely on JSON stringification for simple structural equality.
    // This is sufficient for the current PlayerChoice option shapes and
    // mirrors the behaviour of many lightweight deep-equality helpers.
    try {
      return JSON.stringify(a) === JSON.stringify(b);
    } catch {
      return false;
    }
  }
}
