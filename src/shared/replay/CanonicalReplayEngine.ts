/**
 * CanonicalReplayEngine - Minimal replay engine for parity testing.
 *
 * This engine wraps TurnEngineAdapter in replay mode to provide a clean,
 * coercion-free interface for replaying recorded games. It is designed
 * specifically for parity validation between TypeScript and Python engines.
 *
 * Key design principles:
 * - NO state coercions or host-side hacks
 * - All phase transitions require explicit moves from recordings
 * - Uses the canonical shared orchestrator via TurnEngineAdapter
 * - Thin wrapper with minimal responsibilities
 *
 * This separates parity testing concerns from interactive play concerns,
 * allowing ClientSandboxEngine to focus on UI/UX while this engine handles
 * deterministic replay with strict parity guarantees.
 *
 * @module CanonicalReplayEngine
 */

import type { GameState, Move, GameResult, BoardType, Player, TimeControl } from '../types/game';
import type { SerializedGameState } from '../engine/contracts/serialization';
import {
  TurnEngineAdapter,
  type StateAccessor,
  type DecisionHandler,
  type AdapterMoveResult,
} from '../../server/game/turn/TurnEngineAdapter';
import { hashGameStateSHA256 } from '../engine';
import { evaluateVictory, type VictoryResult } from '../engine';
import { createInitialGameState } from '../engine/initialState';
import { deserializeGameState } from '../engine/contracts/serialization';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Result of applying a single move during replay.
 */
export interface ReplayStepResult {
  /** Whether the move was applied successfully */
  success: boolean;

  /** Game state after applying the move */
  state: GameState;

  /** State hash for parity comparison */
  stateHash: string;

  /** Error message if success is false */
  error?: string;

  /** Whether the game ended after this move */
  isGameOver: boolean;

  /** Victory result if game ended */
  victoryResult?: GameResult;
}

/**
 * Options for creating a CanonicalReplayEngine.
 */
export interface CanonicalReplayEngineOptions {
  /** Game ID for tracking */
  gameId: string;

  /** Board type */
  boardType: BoardType;

  /** Number of players */
  numPlayers: number;

  /** Optional initial state (if not provided, creates fresh state) */
  initialState?: unknown;

  /** Optional debug hook called before/after each move */
  debugHook?: (label: string, state: GameState) => void;
}

/**
 * Summary of the replay engine state for logging.
 */
export interface ReplayStateSummary {
  currentPlayer: number;
  currentPhase: string;
  gameStatus: string;
  moveHistoryLength: number;
  stateHash: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// CANONICAL REPLAY ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * CanonicalReplayEngine - Coercion-free engine for parity testing.
 *
 * This engine is explicitly NOT for interactive play. It provides:
 * - Strict replay semantics with no auto-processing
 * - Canonical state hashing for parity comparison
 * - Direct orchestrator access without host-layer interference
 *
 * Usage:
 * ```typescript
 * const engine = new CanonicalReplayEngine({
 *   gameId: 'test-game-123',
 *   boardType: 'square8',
 *   numPlayers: 2,
 * });
 *
 * for (const move of recordedMoves) {
 *   const result = await engine.applyMove(move);
 *   if (!result.success) {
 *     console.error(`Move failed: ${result.error}`);
 *     break;
 *   }
 *   console.log(`State hash after move: ${result.stateHash}`);
 * }
 * ```
 */
export class CanonicalReplayEngine {
  private currentState: GameState;
  private readonly adapter: TurnEngineAdapter;
  private debugHook: ((label: string, state: GameState) => void) | undefined;
  private appliedMoveCount = 0;

  constructor(options: CanonicalReplayEngineOptions) {
    const { gameId, boardType, numPlayers, initialState, debugHook } = options;

    this.debugHook = debugHook ?? undefined;
    const debugProxy = (label: string, state: GameState) => {
      this.debugHook?.(label, state);
    };

    // Initialize state
    if (initialState && typeof initialState === 'object') {
      const sanitized = this.sanitizeInitialState(initialState);

      // If the caller passed a live GameState (with Map-backed board fields),
      // reuse it directly. Otherwise treat it as a serialized recording.
      const maybeState = sanitized as any;
      const hasLiveBoardMaps =
        maybeState.board &&
        (maybeState.board.stacks instanceof Map ||
          maybeState.board.markers instanceof Map ||
          maybeState.board.collapsedSpaces instanceof Map);

      if (hasLiveBoardMaps) {
        this.currentState = sanitized as GameState;
      } else {
        this.currentState = deserializeGameState(sanitized as SerializedGameState);
      }
    } else {
      // Create fresh initial state
      this.currentState = this.createFreshState(gameId, boardType, numPlayers);
    }

    // Create state accessor that reads/writes to our internal state
    const stateAccessor: StateAccessor = {
      getGameState: () => this.currentState,
      updateGameState: (state) => {
        this.currentState = state;
      },
      getPlayerInfo: (playerNumber) => {
        const player = this.currentState.players.find((p) => p.playerNumber === playerNumber);
        return player ? { type: player.type } : undefined;
      },
    };

    // Decision handler that throws - in replay mode, all decisions come from recordings
    const decisionHandler: DecisionHandler = {
      requestDecision: async (decision) => {
        // This should never be called in proper replay mode
        throw new Error(
          `[CanonicalReplayEngine] Unexpected decision request in replay mode: ${decision.type} for player ${decision.player}. ` +
            `Replay should provide explicit moves for all decisions.`
        );
      },
    };

    // Create adapter in replay mode. Always pass a proxy debugHook so tests can
    // attach or replace hooks after construction without needing to rebuild the adapter.
    this.adapter = new TurnEngineAdapter({
      stateAccessor,
      decisionHandler,
      debugHook: debugProxy,
      replayMode: true,
    });
  }

  /**
   * Attach or replace a debug checkpoint hook.
   *
   * This mirrors GameEngine/ClientSandboxEngine testing utilities and is used
   * by parity suites that want labeled snapshots during replay.
   */
  public setDebugCheckpointHook(hook: (label: string, state: GameState) => void): void {
    this.debugHook = hook;
  }

  /**
   * Emit a synthetic debug checkpoint with the current state.
   */
  public debugCheckpoint(label: string): void {
    this.debugHook?.(label, this.currentState);
  }

  /**
   * Get the current game state (read-only snapshot).
   */
  getState(): Readonly<GameState> {
    return this.currentState;
  }

  /**
   * Get the canonical state hash for parity comparison.
   * Uses SHA-256 hash that matches Python's _compute_state_hash.
   */
  getStateHash(): string {
    return hashGameStateSHA256(this.currentState);
  }

  /**
   * Check if the game has ended.
   */
  isGameOver(): boolean {
    return (
      this.currentState.gameStatus === 'completed' || this.currentState.gameStatus === 'abandoned'
    );
  }

  /**
   * Get a summary of the current state for logging.
   */
  summarize(_label = 'state'): ReplayStateSummary {
    return {
      currentPlayer: this.currentState.currentPlayer,
      currentPhase: this.currentState.currentPhase,
      gameStatus: this.currentState.gameStatus,
      moveHistoryLength: this.currentState.moveHistory.length,
      stateHash: this.getStateHash(),
    };
  }

  /**
   * Get the number of moves applied so far.
   */
  getMoveCount(): number {
    return this.appliedMoveCount;
  }

  /**
   * Apply a move from the recording.
   *
   * In replay mode, the adapter:
   * - Does NOT auto-process single-option decisions
   * - Does NOT auto-process line rewards
   * - Breaks immediately when a decision is required
   *
   * The caller must provide explicit moves for every phase transition,
   * exactly as they appear in the recording.
   */
  async applyMove(move: Move): Promise<ReplayStepResult> {
    this.debugHook?.(`before-applyMove-${this.appliedMoveCount + 1}`, this.currentState);

    try {
      const result: AdapterMoveResult = await this.adapter.processMove(move);

      this.appliedMoveCount += 1;

      if (!result.success) {
        return {
          success: false,
          state: this.currentState,
          stateHash: this.getStateHash(),
          error: result.error || 'Unknown error',
          isGameOver: this.isGameOver(),
        };
      }

      // IMPORTANT: Append the move to moveHistory for accurate phase transition
      // logic. The orchestrator's computeHadAnyActionThisTurn relies on moveHistory
      // to decide whether to enter forced_elimination phase. Without this, replay
      // would incorrectly think no actions were taken this turn.
      // DEBUG: trace mustMoveFromStackKey
      if (process.env.RINGRIFT_TRACE_DEBUG === '1' && move.type === 'place_ring') {
        // eslint-disable-next-line no-console
        console.log(
          '[CanonicalReplayEngine] BEFORE spread, mustMoveFromStackKey:',
          this.currentState.mustMoveFromStackKey
        );
      }
      this.currentState = {
        ...this.currentState,
        moveHistory: [...this.currentState.moveHistory, move],
      };
      if (process.env.RINGRIFT_TRACE_DEBUG === '1' && move.type === 'place_ring') {
        // eslint-disable-next-line no-console
        console.log(
          '[CanonicalReplayEngine] AFTER spread, mustMoveFromStackKey:',
          this.currentState.mustMoveFromStackKey
        );
      }

      this.debugHook?.(`after-applyMove-${this.appliedMoveCount}`, this.currentState);

      // Check for victory via orchestrator result or explicit evaluation
      const victoryResult = result.victoryResult || this.evaluateVictoryIfNeeded();

      const stepResult: ReplayStepResult = {
        success: true,
        state: this.currentState,
        stateHash: this.getStateHash(),
        isGameOver: this.isGameOver() || victoryResult !== undefined,
      };
      if (victoryResult !== undefined) {
        stepResult.victoryResult = victoryResult;
      }
      return stepResult;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);

      return {
        success: false,
        state: this.currentState,
        stateHash: this.getStateHash(),
        error: errorMessage,
        isGameOver: this.isGameOver(),
      };
    }
  }

  /**
   * Get the victory result if the game has ended.
   */
  getVictoryResult(): GameResult | null {
    if (!this.isGameOver()) {
      return null;
    }

    const verdict = evaluateVictory(this.currentState);
    if (!verdict.isGameOver) {
      return null;
    }

    return this.convertVictoryResultToGameResult(verdict);
  }

  /**
   * Create a fresh initial game state.
   */
  private createFreshState(gameId: string, boardType: BoardType, numPlayers: number): GameState {
    const timeControl: TimeControl = {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    };

    const players: Player[] = [];
    for (let i = 0; i < numPlayers; i += 1) {
      const playerNumber = i + 1;
      players.push({
        id: `player-${playerNumber}`,
        username: `Player ${playerNumber}`,
        type: 'ai',
        playerNumber,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      });
    }

    return createInitialGameState(gameId, boardType, players, timeControl, false);
  }

  /**
   * Sanitize initial state from recording (clear history arrays).
   */
  private sanitizeInitialState(rawState: unknown): unknown {
    const state = rawState as Record<string, unknown>;
    const sanitized = { ...state };

    // Clear any existing history to ensure clean replay
    if (Array.isArray(sanitized.moveHistory)) {
      sanitized.moveHistory = [];
    }
    if (Array.isArray(sanitized.history)) {
      sanitized.history = [];
    }

    return sanitized;
  }

  /**
   * Evaluate victory if the game appears to be in a terminal state.
   */
  private evaluateVictoryIfNeeded(): GameResult | undefined {
    // Fast path: explicit terminal status
    if (
      this.currentState.gameStatus !== 'completed' &&
      this.currentState.gameStatus !== 'abandoned'
    ) {
      // Additional terminal guard for ANM/LPS: if the board is bare (no stacks)
      // and all players have zero rings in hand, apply stalemate detection.
      // When stacks exist, the game is NOT terminal - players with stacks can
      // always act (via movement/capture or forced elimination per RR-CANON-R072).
      const hasStacks = this.currentState.board.stacks.size > 0;
      if (!hasStacks) {
        const allZeroRings = this.currentState.players.every((p) => p.ringsInHand <= 0);
        if (allZeroRings) {
          // Bare board with no rings in hand - apply stalemate tiebreakers.
          this.currentState = {
            ...this.currentState,
            currentPhase: 'game_over',
            gameStatus: 'completed',
          };
        }
      }
    }

    return this.getVictoryResult() ?? undefined;
  }

  /**
   * Convert VictoryResult (from evaluateVictory) to GameResult format.
   * Builds scores from current game state since VictoryResult doesn't include them.
   */
  private convertVictoryResultToGameResult(victory: VictoryResult): GameResult | null {
    if (!victory.isGameOver || victory.winner === undefined) {
      return null;
    }

    const ringsEliminated: Record<number, number> = {};
    const territorySpaces: Record<number, number> = {};
    const ringsRemaining: Record<number, number> = {};

    // Build scores from current state
    for (const player of this.currentState.players) {
      ringsEliminated[player.playerNumber] = player.eliminatedRings ?? 0;
      territorySpaces[player.playerNumber] = player.territorySpaces ?? 0;
      ringsRemaining[player.playerNumber] = player.ringsInHand ?? 0;
    }

    // Count rings on board per player
    for (const stack of this.currentState.board.stacks.values()) {
      const controlling = stack.controllingPlayer;
      if (controlling !== undefined) {
        ringsRemaining[controlling] = (ringsRemaining[controlling] || 0) + stack.stackHeight;
      }
    }

    // Map orchestrator reasons to game result reasons
    const reasonMap: Record<string, GameResult['reason']> = {
      ring_elimination: 'ring_elimination',
      territory_control: 'territory_control',
      last_player_standing: 'last_player_standing',
      game_completed: 'game_completed',
    };

    return {
      winner: victory.winner,
      reason: reasonMap[victory.reason || ''] || 'game_completed',
      finalScore: {
        ringsEliminated,
        territorySpaces,
        ringsRemaining,
      },
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FACTORY FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a CanonicalReplayEngine from a game detail object.
 *
 * Convenience factory for use with SelfPlayGameService.getGame() results.
 */
export function createCanonicalReplayEngine(
  gameId: string,
  boardType: BoardType,
  numPlayers: number,
  initialState?: unknown,
  debugHook?: (label: string, state: GameState) => void
): CanonicalReplayEngine {
  const options: CanonicalReplayEngineOptions = {
    gameId,
    boardType,
    numPlayers,
  };
  if (initialState !== undefined) {
    options.initialState = initialState;
  }
  if (debugHook !== undefined) {
    options.debugHook = debugHook;
  }
  return new CanonicalReplayEngine(options);
}
