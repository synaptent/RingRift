/**
 * AI Fallback Handler - Shared utilities for AI fallback move selection
 *
 * This module provides shared utilities for handling AI fallback scenarios
 * when the primary AI service is unavailable or fails. Both server (GameSession)
 * and client (sandbox) use these utilities for consistent fallback behavior.
 *
 * The core move selection logic is in `localAIMoveSelection.ts`. This module
 * provides:
 * - Type definitions for fallback contexts
 * - Telemetry/diagnostics helpers
 * - Utility functions for common fallback patterns
 *
 * Usage:
 * ```typescript
 * import { selectFallbackMove, FallbackContext } from './AIFallbackHandler';
 *
 * const context: FallbackContext = {
 *   reason: 'service_timeout',
 *   playerNumber: 2,
 *   gameState: state,
 *   validMoves: moves,
 *   rng: createLocalAIRng(),
 * };
 *
 * const move = selectFallbackMove(context);
 * ```
 *
 * @module AIFallbackHandler
 */

import type { GameState, Move } from '../types/game';
import {
  chooseLocalMoveFromCandidates,
  LocalAIRng,
} from '../engine/localAIMoveSelection';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Reasons why fallback was triggered.
 */
export type FallbackReason =
  | 'service_unavailable'
  | 'service_timeout'
  | 'service_error'
  | 'no_move_returned'
  | 'move_rejected'
  | 'explicit_local_mode';

/**
 * Context for fallback move selection.
 */
export interface FallbackContext {
  /** Why fallback was triggered */
  reason: FallbackReason;
  /** Player needing a move */
  playerNumber: number;
  /** Current game state */
  gameState: GameState;
  /** Valid moves for the player */
  validMoves: Move[];
  /** RNG for deterministic selection */
  rng: LocalAIRng;
  /** Optional randomness factor for training */
  randomness?: number;
}

/**
 * Result of fallback move selection.
 */
export interface FallbackResult {
  /** Selected move (null if no valid moves) */
  move: Move | null;
  /** Whether selection succeeded */
  success: boolean;
  /** Reason for selection */
  reason: FallbackReason;
  /** Diagnostics for telemetry */
  diagnostics: FallbackDiagnostics;
}

/**
 * Diagnostics for fallback operations.
 */
export interface FallbackDiagnostics {
  /** Number of valid moves available */
  validMoveCount: number;
  /** Move type that was selected (if any) */
  selectedMoveType?: string;
  /** Time taken for selection (ms) */
  selectionTimeMs: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// FALLBACK SELECTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Select a fallback move using the local heuristic policy.
 *
 * This is a thin wrapper around `chooseLocalMoveFromCandidates` that adds
 * diagnostics and consistent error handling for use by both server and client.
 *
 * @param context Fallback context with game state and valid moves
 * @returns Fallback result with selected move and diagnostics
 */
export function selectFallbackMove(context: FallbackContext): FallbackResult {
  const startTime = performance.now();

  const move = chooseLocalMoveFromCandidates(
    context.playerNumber,
    context.gameState,
    context.validMoves,
    context.rng,
    context.randomness ?? 0
  );

  const selectionTimeMs = performance.now() - startTime;

  return {
    move,
    success: move !== null,
    reason: context.reason,
    diagnostics: {
      validMoveCount: context.validMoves.length,
      selectedMoveType: move?.type,
      selectionTimeMs,
    },
  };
}

/**
 * Check if fallback should be attempted based on context.
 *
 * @param validMoves Available valid moves
 * @param gameState Current game state
 * @returns Whether fallback selection makes sense
 */
export function shouldAttemptFallback(validMoves: Move[], gameState: GameState): boolean {
  // No point attempting fallback if no valid moves
  if (!validMoves || validMoves.length === 0) {
    return false;
  }

  // Don't fallback if game is over
  if (gameState.gameStatus === 'completed') {
    return false;
  }

  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// RNG UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a local AI RNG from a seed.
 *
 * This creates a deterministic RNG suitable for fallback move selection.
 * For full reproducibility, the seed should be derived from the game state's
 * rngSeed combined with the player number and move count.
 *
 * @param seed RNG seed value
 * @returns LocalAIRng function
 */
export function createLocalAIRng(seed: number): LocalAIRng {
  // Simple mulberry32 PRNG for consistent behavior across environments
  let state = seed;

  return (): number => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Derive a fallback RNG seed from game state.
 *
 * This creates a deterministic seed for fallback move selection that is
 * reproducible across server and client for a given game position.
 *
 * @param gameState Current game state
 * @param playerNumber Player needing the move
 * @returns Seed for RNG creation
 */
export function deriveFallbackSeed(gameState: GameState, playerNumber: number): number {
  const baseSeed = gameState.rngSeed ?? 0;
  const moveCount = gameState.moveHistory?.length ?? 0;

  // Combine base seed with player and move count for uniqueness
  return (baseSeed * 31 + playerNumber * 17 + moveCount * 13) >>> 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// DIAGNOSTICS TRACKING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Cumulative fallback diagnostics for a session/game.
 */
export interface CumulativeFallbackDiagnostics {
  /** Total fallback attempts */
  totalAttempts: number;
  /** Successful selections */
  successfulSelections: number;
  /** Failed selections (no valid moves) */
  failedSelections: number;
  /** Breakdown by reason */
  byReason: Partial<Record<FallbackReason, number>>;
  /** Total selection time (ms) */
  totalSelectionTimeMs: number;
}

/**
 * Create empty cumulative diagnostics.
 */
export function createEmptyDiagnostics(): CumulativeFallbackDiagnostics {
  return {
    totalAttempts: 0,
    successfulSelections: 0,
    failedSelections: 0,
    byReason: {},
    totalSelectionTimeMs: 0,
  };
}

/**
 * Update cumulative diagnostics with a new result.
 *
 * @param cumulative Existing cumulative diagnostics
 * @param result New fallback result to incorporate
 * @returns Updated cumulative diagnostics
 */
export function updateDiagnostics(
  cumulative: CumulativeFallbackDiagnostics,
  result: FallbackResult
): CumulativeFallbackDiagnostics {
  return {
    totalAttempts: cumulative.totalAttempts + 1,
    successfulSelections: cumulative.successfulSelections + (result.success ? 1 : 0),
    failedSelections: cumulative.failedSelections + (result.success ? 0 : 1),
    byReason: {
      ...cumulative.byReason,
      [result.reason]: (cumulative.byReason[result.reason] ?? 0) + 1,
    },
    totalSelectionTimeMs: cumulative.totalSelectionTimeMs + result.diagnostics.selectionTimeMs,
  };
}
